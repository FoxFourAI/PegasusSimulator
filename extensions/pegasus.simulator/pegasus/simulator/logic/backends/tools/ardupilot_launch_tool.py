"""
| File: ardupilot_launch_tool.py
| Author: Tomer Tip (tomerT1212@gmail.com)
| Editor: Nataliia Yurevych (nataliia.yurevych@foxfour.ai)
| License: BSD-3-Clause. Copyright (c) 2024, Tomer Tip. All rights reserved.
| Description: Defines an auxiliary tool to launch the Ardupilot process in the background
"""

# System tools used to launch the ardupilot process in the brackground
import os
import signal
import tempfile
import subprocess
import psutil


class ArduPilotLaunchTool:
    """
    A class that manages the start/stop of an ardupilot process. It requires only the path to the Ardupilot installation (assuming that
    Ardupilot was already built with 'make ardupilot_sitl_default none'), the vehicle id and the vehicle model.
    """

    def __init__(self, ardupilot_dir, vehicle_id: int = 0, ardupilot_model: str = "gazebo-iris"):
        """Construct the ArduPilotLaunchTool object

        Args:
            ardupilot_dir (str): A string with the path to the Ardupilot-Autopilot directory
            vehicle_id (int): The ID of the vehicle. Defaults to 0.
            ardupilot_model (str): The vehicle model. Defaults to "iris".
        """

        # Attribute that will hold the ardupilot process once it is running
        self.ardupilot_process = None

        # The vehicle id (used for the mavlink port open in the system)
        self.vehicle_id = vehicle_id

        # Configurations to whether autostart ardupilot (SITL) automatically or have the user launch it manually on another
        # terminal
        self.ardupilot_dir = ardupilot_dir

        # Ardupilot frame
        self.ardupilot_model = ardupilot_model

        # Ardupilot FDM communication type:
        self.model = "JSON"

        # Create a temporary filesystem for ardupilot to write data to/from (and modify the origin rcS files)
        self.root_fs = tempfile.TemporaryDirectory()

        # Set the environement variables that let Ardupilot know which vehicle model to use internally
        # Use clean system environment instead of Isaac Sim's environment
        self.environment = os.environ.copy()

        # Remove Isaac Sim's Python paths that cause conflicts
        if 'PYTHONPATH' in self.environment:
            # Filter out Isaac Sim Python paths
            python_paths = self.environment['PYTHONPATH'].split(':')
            filtered_paths = [path for path in python_paths if 'isaacsim' not in path.lower()]
            if filtered_paths:
                self.environment['PYTHONPATH'] = ':'.join(filtered_paths)
            else:
                del self.environment['PYTHONPATH']

        # Ensure we use system Python
        if 'PATH' in self.environment:
            # Make sure system paths come first
            system_paths = ['/usr/bin', '/usr/local/bin', '/bin']
            current_paths = self.environment['PATH'].split(':')
            # Remove Isaac Sim paths and put system paths first
            filtered_paths = [path for path in current_paths if 'isaacsim' not in path.lower()]
            new_path = ':'.join(system_paths + filtered_paths)
            self.environment['PATH'] = new_path

    def _sitl_already_exists(self):
        return os.path.exists(f'{self.ardupilot_dir}/build/sitl/bin/arducopter')

    def _get_vehicle_frame(self):
        return self.ardupilot_model

    def launch_ardupilot(self):
        """
        Method that will launch an ardupilot instance with the specified configuration
        """
        # sim_vehicle.py -v ArduCopter -f gazebo-iris --mode JSON --console --map
        command = [
            "/usr/bin/python3", f"{self.ardupilot_dir}/Tools/autotest/sim_vehicle.py",
            "-v", "ArduCopter",
            "-f", f"{self._get_vehicle_frame()}",
            "--model", f"{self.model}",
            f"{'--no-rebuild' if self._sitl_already_exists() else ''}",
            f"--console",
            f"--map",
            "-I", f"{self.vehicle_id}",
            "--sysid", f"{self.vehicle_id + 1}",
            "--out", f"udp:127.0.0.1:{14550 + self.vehicle_id * 10}",
        ]

        # Filter out empty strings from the command
        command = [arg for arg in command if arg.strip()]
        command_str = " ".join(command)

        print(f"Launching ArduPilot with command: {command_str}")

        # Try different terminal approaches
        terminal_commands = [
            # First try: gnome-terminal with explicit system python
            ["/usr/bin/gnome-terminal", "--", "/bin/bash", "-c",
             f"cd {self.root_fs.name} && exec {command_str}"],
        ]

        # Try each terminal command until one works
        for terminal_cmd in terminal_commands:
            try:
                # Check if the terminal exists
                if os.path.exists(terminal_cmd[0]):
                    print(f"Trying terminal: {terminal_cmd[0]}")
                    self.ardupilot_process = subprocess.Popen(
                        terminal_cmd,
                        cwd=self.root_fs.name,
                        env=self.environment,
                        preexec_fn=os.setsid,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )

                    # Wait a moment to see if it crashes immediately
                    import time
                    time.sleep(1)

                    if self.ardupilot_process.poll() is None:
                        print(f"Successfully launched ArduPilot using {terminal_cmd[0]}")
                        return
                    else:
                        print(f"Terminal {terminal_cmd[0]} failed to launch ArduPilot")
                        continue

            except Exception as e:
                print(f"Failed to launch with {terminal_cmd[0]}: {e}")
                continue

        # If all terminals failed, try without terminal (background process)
        print("All terminal methods failed, trying background process...")
        try:
            self.ardupilot_process = subprocess.Popen(
                command,
                cwd=self.root_fs.name,
                env=self.environment,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("ArduPilot launched as background process (no terminal UI)")
        except Exception as e:
            print(f"Failed to launch ArduPilot: {e}")
            raise

    def kill_ardupilot(self):
        """
        Method that will kill a ardupilot instance with the specified configuration
        """
        if self.ardupilot_process is not None:
            try:
                os.killpg(self.ardupilot_process.pid, signal.SIGINT)
            except:
                pass
            try:
                self.ardupilot_process.kill()
                self.ardupilot_process.wait()
            except:
                pass
            self.ardupilot_process = None

        # Define the keywords to search for in process names
        keywords = ['arducopter', 'mavproxy']

        # Get the list of all running processes using the ps command
        try:
            ps_output = subprocess.run(['ps', '-aux'], capture_output=True, text=True)

            # Filter processes that match any of the keywords (case insensitive)
            matching_processes = [
                line for line in ps_output.stdout.splitlines()
                if any(keyword in line.lower() for keyword in keywords)
            ]

            # Extract process IDs (PID) from the filtered lines
            pids = [line.split()[1] for line in matching_processes]

            # Kill each matching process by its PID
            for pid in pids:
                try:
                    os.kill(int(pid), 9)
                    print(f"Killed process {pid}")
                except ProcessLookupError:
                    print(f"Process {pid} not found.")
                except PermissionError:
                    print(f"Permission denied to kill process {pid}.")
                except Exception as e:
                    print(f"Failed to kill process {pid}: {e}")
        except Exception as e:
            print(f"Error during process cleanup: {e}")

    def __del__(self):
        """
        If the ardupilot process is still running when the Ardupilot launch tool object is whiped from memory, then make sure
        we kill the ardupilot instance so we don't end up with hanged ardupilot instances
        """

        # Make sure the Ardupilot process gets killed
        if self.ardupilot_process:
            self.kill_ardupilot()

        # Make sure we clean the temporary filesystem used for the simulation
        if hasattr(self, 'root_fs'):
            self.root_fs.cleanup()


# ---- Code used for debugging the ardupilot tool ----
def main():

    ardupilot_tool = ArduPilotLaunchTool(os.environ["HOME"] + "/ardupilot")
    print("Launching ArduPilot")
    ardupilot_tool.launch_ardupilot()

    import time

    # print("Killing ArduPilot in 5")
    time.sleep(3000)
    # ardupilot_tool.kill_ardupilot()


if __name__ == "__main__":
    main()