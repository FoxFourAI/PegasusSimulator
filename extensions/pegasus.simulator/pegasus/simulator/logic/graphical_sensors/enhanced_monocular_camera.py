"""
| File: enhanced_monocular_camera.py
| Author: Nataliia Yurevych (nataliia.yurevych@foxfour.ai)
| Description: Enhanced MonocularCamera with H.264 streaming. Simulates a monocular camera attached to the vehicle.
"""
__all__ = ["EnhancedMonocularCamera"]

import cv2
import numpy as np
import time
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.udp_h264_streamer import H264RTPStreamer
from pegasus.simulator.logic.state import State


class EnhancedMonocularCamera(MonocularCamera):
    """
    The class that implements an enhanced monocular camera sensor with H.264 streaming. This class inherits from the
    MonocularCamera which inherits from GraphicalSensor.
    """

    def __init__(self, camera_name: str, config: dict = {}):
        """
        Initialize the EnhancedMonocularCamera class

        Check the oficial documentation for the Camera class in Isaac Sim:
        https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html#isaac-sim-sensors-camera

        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the MonocularCamera - it can be empty or only have some of the parameters used by the EnhancedMonocularCamera.

        Examples:
            The dictionary default parameters are

            >>> {"frequency": 30,
            >>> "position": np.array([0.30, 0.0, 0.0]),
            >>> "orientation": np.array([0.0, 0.0, 0.0]),
            >>> "resolution": (1920, 1200),
            >>> "udp_streaming": False,
            >>> "udp_host": "127.0.0.1",
            >>> "udp_port": 8081,
            >>> "stream_width": 640,
            >>> "stream_height": 480,
            >>> "stream_fps": 30,
            >>> "stream_bitrate": 1000000,  # 1 Mbps
            >>> "debug_mode": True,
            >>> "test_pattern_mode": False,
            >>> "depth": True,
            >>> "intrinsics": np.array([[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]),
            >>> "distortion_coefficients": np.array([0.14, -0.03, -0.0002, -0.00003, 0.009, 0.5, -0.07, 0.017]),
            >>> "diagonal_fov": 140.0}
        """

        # Initialize the Super class "object" attributes
        super().__init__(camera_name, config)

        # UDP configuration
        self.udp_enabled = config.get("udp_streaming", False)
        self.udp_config = config
        self.udp_streamer = None

        # Frame processing control
        self.frame_counter = 0
        self.udp_frame_skip = max(1, int(60.0 / config.get("frequency", 30.0))) # How often a frame is sent over UDP for H.264 streaming assuming the simulation is running 60 FPS

        # Debug and initialization tracking
        self.debug_mode = config.get("debug_mode", True)
        self.test_pattern_mode = config.get("test_pattern_mode", False)
        self.camera_ready = False
        self.initialization_frames = 0
        self.max_init_frames = 25

        # H.264 streaming parameters
        self.stream_width = config.get("stream_width", 640)
        self.stream_height = config.get("stream_height", 480)
        self.stream_fps = config.get("stream_fps", 30)
        self.stream_bitrate = config.get("stream_bitrate", 2000000)

        # Stall detection (when Isaac Sim stops delivering valid camera frames)
        self.last_valid_frame = None
        self.consecutive_none_frames = 0
        self.last_successful_frame_time = time.time()

        print(f"Enhanced camera initialized: {camera_name}")

    def initialize(self, vehicle):
        """Initialize camera with UDP streaming"""
        super().initialize(vehicle)

        if self.debug_mode:
            print(f"[DEBUG] Initializing camera: {self._camera_name}")

        if self.udp_enabled:
            self.udp_streamer = H264RTPStreamer(
                host=self.udp_config.get("udp_host", "127.0.0.1"),
                port=self.udp_config.get("udp_port", 8081),
                fps=self.stream_fps,
                bitrate=self.stream_bitrate // 1000,
                width=self.stream_width,
                height=self.stream_height,
                debug_output=self.debug_mode
            )

            if self.debug_mode:
                print(f"[DEBUG] H.264 Streamer: {self.udp_streamer.host}:{self.udp_streamer.port}")


    def start(self):
        """Start camera"""
        super().start()

        if self.debug_mode:
            print(f"[DEBUG] Starting enhanced camera: {self._camera_name}")

        if self.udp_streamer:
            success = self.udp_streamer.start()
            if self.debug_mode:
                status = "SUCCESS" if success != False else "FAILED"
                print(f"[DEBUG] H.264 streaming: {status}")

        self.camera_ready = False
        self.initialization_frames = 0
        self.last_successful_frame_time = time.time()


    def stop(self):
        """Stop camera with cleanup"""
        if self.udp_streamer:
            self.udp_streamer.stop()
            if self.debug_mode:
                print(f"[DEBUG] H.264 streaming stopped for {self._camera_name}")

        super().stop()

    def wait_for_camera_ready(self):
        """Camera ready check"""
        if self.camera_ready:
            return True

        self.initialization_frames += 1

        if self.initialization_frames < self.max_init_frames:
            if self.debug_mode and self.initialization_frames % 25 == 0:
                print(f"[DEBUG] Camera init: {self.initialization_frames}/{self.max_init_frames}")
            return False

        self.camera_ready = True
        if self.debug_mode:
            print(f"[DEBUG] Camera ready after {self.initialization_frames} frames")
        return True

    def get_isaac_camera_data(self):
        """Get RGB data from Isaac Sim camera"""
        isaac_camera = self._camera

        current_frame = isaac_camera.get_current_frame()
        key = 'rgb' # It actually returns RGBA so we have to get rid of alpha channel
        data = current_frame[key]

        if self.debug_mode:
            print(f"[DEBUG] Processing: {data.shape}, range: [{data.min()}, {data.max()}]")

        rgb_data = data[:, :, :3]  # Drop alpha channel

        return rgb_data

    def create_simple_test_pattern(self):
        """Create simple test pattern"""
        width = self.stream_width
        height = self.stream_height

        # Gray background
        test_frame = np.full((height, width, 3), (80, 80, 80), dtype=np.uint8)

        # Add a large frame number in the frame center
        frame_num = self.frame_counter % 1000
        cv2.putText(test_frame, f"{frame_num}", (width//2 - 60, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

        # Add the status in the top left corner of the frame
        cv2.putText(test_frame, "INIT", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return test_frame

    def update(self, state: State, dt: float):
        """Update method"""
        # Call parent update
        camera_data = super().update(state, dt)

        # Process frame for streaming
        if self.udp_streamer and self._camera_full_set:
            self.process_frame_for_streaming(camera_data)

        return camera_data

    def process_frame_for_streaming(self, camera_data):
        """Frame processing"""
        # Frame skip logic to maintain the correct frame rate
        if self.frame_counter % self.udp_frame_skip != 0:
            self.frame_counter += 1
            return

        frame_source = "unknown"

        # During initialization
        if not self.wait_for_camera_ready():
            rgb_frame = self.create_simple_test_pattern()
            frame_source = "init_pattern"

        # Force test pattern mode If manually turned on test pattern mode
        elif self.test_pattern_mode:
            rgb_frame = self.create_simple_test_pattern()
            frame_source = "forced_pattern"

        # Get real camera data
        else:
            rgb_frame = self.get_isaac_camera_data()

            self.last_valid_frame = rgb_frame.copy()
            self.last_successful_frame_time = time.time()
            frame_source = "camera_data"

        # Convert RGB to BGR for streaming
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Stream the frame
        self.udp_streamer.add_frame(bgr_frame)

        if self.debug_mode:
            print(f"[DEBUG] Encoded & streamed {frame_source}: {bgr_frame.shape}")

        self.frame_counter += 1

    def set_test_pattern_mode(self, enabled: bool):
        """Enable/disable test pattern mode"""
        self.test_pattern_mode = enabled
        if self.debug_mode:
            mode = "enabled" if enabled else "disabled"
            print(f"[DEBUG] Test pattern mode: {mode}")

    def set_debug_mode(self, enabled: bool):
        """Enable/disable debug output"""
        self.debug_mode = enabled

    def get_streaming_info(self):
        """Get streaming information"""
        if not self.udp_streamer:
            return None

        return {
            "enabled": self.udp_enabled,
            "host": getattr(self.udp_streamer, 'host', 'unknown'),
            "port": getattr(self.udp_streamer, 'port', 'unknown'),
            "fps": getattr(self.udp_streamer, 'fps', 'unknown'),
            "resolution": f"{self.stream_width}x{self.stream_height}",
            "bitrate": getattr(self.udp_streamer, 'bitrate', 'unknown'),
            "running": getattr(self.udp_streamer, 'running', False),
            "consecutive_none_frames": self.consecutive_none_frames
        }
