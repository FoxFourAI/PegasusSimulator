"""
| File: enhanced_monocular_camera.py
| Author: Nataliia Yurevych (nataliia.yurevych@foxfour.ai)
| Description: Enhanced MonocularCamera with H.264 streaming. Simulates a monocular camera attached to the vehicle.
"""
__all__ = ["EnhancedMonocularCamera"]

import cv2
import numpy as np
import time
import threading
import queue

from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.udp_h264_streamer import H264RTPStreamer
from pegasus.simulator.logic.state import State

# ROS 2 Imports (Safety check)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[WARNING] ROS 2 not found. Depth publishing will be disabled.")


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
            >>> "stream_bitrate": 2000000,  # 2 Mbps
            >>> "debug_mode": True,
            >>> "test_pattern_mode": False,
            >>> "depth": True,
            >>> "intrinsics": np.array([[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]),
            >>> "distortion_coefficients": np.array([0.14, -0.03, -0.0002, -0.00003, 0.009, 0.5, -0.07, 0.017]),
            >>> "diagonal_fov": 140.0}
        """

        # Ensure depth is enabled in the configuration
        if "depth" not in config:
            config["depth"] = True

        # Initialize the Super class "object" attributes
        super().__init__(camera_name, config)

        # -------------------------
        # ROS 2 Depth Streaming Setup
        # -------------------------
        self.ros2_enabled = config.get("ros2_depth_streaming", False) and ROS2_AVAILABLE
        self.ros2_topic = config.get("ros2_depth_topic", f"/{camera_name}/depth")
        self.ros2_queue_size = config.get("ros2_queue_size", 2)

        # Threading setup to avoid blocking simulation physics
        self.depth_queue = queue.Queue(maxsize=self.ros2_queue_size)
        self.stop_threads = False
        self.ros_node = None
        self.depth_pub = None

        if self.ros2_enabled:
            # Initialize ROS 2 logic
            self._init_ros2_node(camera_name)

            # Start worker thread
            self.worker_thread = threading.Thread(target=self._ros_worker, daemon=True)
            self.worker_thread.start()
            print(f"ROS 2 Depth Streamer initialized on topic: {self.ros2_topic}")

        # UDP configuration
        self.udp_enabled = config.get("udp_streaming", False)
        self.udp_config = config
        self.udp_streamer = None

        # Frame processing control
        self.frame_counter = 0
        sensor_frequency = config.get("frequency", 30.0)
        desired_stream_fps = config.get("stream_fps", 30)
        self.udp_frame_skip = max(1, int(60 / desired_stream_fps))

        # Debug and initialization tracking
        self.debug_mode = config.get("debug_mode", True)
        self.test_pattern_mode = config.get("test_pattern_mode", False)
        self.camera_ready = False
        self.initialization_frames = 0
        self.max_init_frames = 5

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

    def _init_ros2_node(self, camera_name):
        """Helper to initialize ROS node and publisher safely"""
        try:
            if not rclpy.ok():
                rclpy.init()

            # Create a unique node for this camera
            node_name = f"{camera_name}_depth_publisher_node"
            self.ros_node = rclpy.create_node(node_name)

            # Create Publisher for sensor_msgs/Image
            self.depth_pub = self.ros_node.create_publisher(Image, self.ros2_topic, 10)

        except Exception as e:
            print(f"[ERROR] Failed to init ROS 2 node: {e}")
            self.ros2_enabled = False

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
        data = current_frame['rgb'] # It actually returns RGBA so we have to get rid of the alpha channel

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
        # Call parent update to ensure sensors are ticked
        camera_data = super().update(state, dt)

        if self._camera_full_set:
            # 1. Handle H.264 Video Streaming
            if self.udp_streamer:
                self.process_frame_for_streaming(camera_data)

            # 2. Handle ROS 2 Depth Streaming
            if self.ros2_enabled:
                self.process_and_queue_depth()

        return camera_data

    def process_and_queue_depth(self):
        """
        Extracts depth data, clips it, and queues it for ROS publishing.
        """
        current_frame = self._camera.get_current_frame()

        # --- FIX: Check for the key that ACTUALLY exists ---
        if "distance_to_image_plane" not in current_frame:
            # Fallback logic in case the sensor isn't ready
            return

        # --- FIX: Use 'distance_to_image_plane' (Z-depth) instead of 'distance_to_camera' ---
        depth_map = current_frame["distance_to_image_plane"]

        # 2. FIX: Check if the data itself is None (happens during sim startup)
        if depth_map is None:
            return

        # Optimize: Only process if the queue has space (drop frame otherwise)
        if not self.depth_queue.full():
            # 1. Clean Data: Replace Infinite values (Sky is usually inf)
            depth_map = np.nan_to_num(depth_map, posinf=6.0, neginf=0.0)

            # 2. Clip Logic: 0.2m to 6.0m
            processed_depth = np.clip(depth_map, 0.2, 6.0)

            # 3. Queue the data along with the current time
            timestamp = time.time()
            self.depth_queue.put((processed_depth, timestamp))

            # (Optional) Debug Print to confirm data is flowing
            if self.frame_counter % 60 == 0:
               print(f"Queueing Depth: {processed_depth.shape}")

    def _ros_worker(self):
        """
        Background thread that picks depth arrays from queue and publishes ROS messages.
        """
        while not self.stop_threads:
            try:
                # Wait for data with a timeout
                item = self.depth_queue.get(timeout=1.0)
                depth_data, timestamp = item
            except queue.Empty:
                continue

            try:
                # Construct sensor_msgs/Image manually (faster than cv_bridge dependency)
                msg = Image()

                # Header
                msg.header = Header()
                msg.header.stamp = self.ros_node.get_clock().now().to_msg()
                msg.header.frame_id = f"{self._camera_name}_optical_frame"

                # Dimensions
                height, width = depth_data.shape
                msg.height = height
                msg.width = width

                # Encoding: 32-bit Floating Point, Single Channel (Depth)
                msg.encoding = "32FC1"
                msg.is_bigendian = 0
                msg.step = width * 4  # 4 bytes per float32

                # Data Payload
                # Ensure it is float32 before converting to bytes
                msg.data = depth_data.astype(np.float32).tobytes()

                # Publish
                self.depth_pub.publish(msg)

            except Exception as e:
                if self.debug_mode:
                    print(f"[ROS ERROR] Failed to publish depth: {e}")
            finally:
                self.depth_queue.task_done()

    def process_frame_for_streaming(self, camera_data):
        """Frame processing"""
        # Frame skip logic to maintain the correct frame rate
        if self.frame_counter % self.udp_frame_skip != 0:
            self.frame_counter += 1
            return

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
