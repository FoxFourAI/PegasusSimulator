try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
    from geometry_msgs.msg import Point32, TransformStamped
    import struct
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
    from tf2_ros import TransformBroadcaster
    ROS2_AVAILABLE = True
except ImportError as e:
    print(f"ROS2 is not available in the main file.\n{e}")
    import time
    while (True):
        print("No tf")
        time.sleep(1000)
    ROS2_AVAILABLE = False

class WorldToCameraTransformPublisher(Node):
    """
    ROS2 Node that publishes the transform from world frame to camera frame.
    This computes the combined transform: world -> drone -> camera = world -> camera
    """

    def __init__(self, camera_config: dict, world_frame_id: str = "world", camera_frame_id: str = "camera"):
        super().__init__('world_to_camera_tf_publisher')

        self.world_frame_id = world_frame_id
        self.camera_frame_id = camera_frame_id

        # Dynamic transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Extract camera pose relative to drone body from config
        self.camera_position_in_drone = camera_config.get("position", np.array([0.0, 0.0, 0.0]))
        self.camera_orientation_euler = camera_config.get("orientation", np.array([0.0, 0.0, 0.0]))

        # Convert euler angles (degrees) to rotation matrix for camera relative to drone
        # The orientation is given as [roll, pitch, yaw] in degrees
        self.R_drone_to_camera = Rotation.from_euler('XYZ', self.camera_orientation_euler, degrees=True)

        self.get_logger().info(
            f"World-to-Camera TF Publisher initialized:\n"
            f"  Camera position (in drone frame): {self.camera_position_in_drone}\n"
            f"  Camera orientation (euler deg): {self.camera_orientation_euler}"
        )

    def compute_world_to_camera_transform(self, drone_position: np.ndarray, drone_orientation_quat: np.ndarray):
        """
        Compute the world-to-camera transform by combining:
        T_world_to_camera = T_world_to_drone * T_drone_to_camera

        Args:
            drone_position: [x, y, z] position of drone in world frame
            drone_orientation_quat: [x, y, z, w] quaternion of drone orientation in world frame

        Returns:
            camera_position_world: [x, y, z] camera position in world frame
            camera_orientation_quat: [x, y, z, w] camera orientation quaternion in world frame
        """
        # Rotation from world to drone
        R_world_to_drone = Rotation.from_quat(drone_orientation_quat)

        # Transform camera position from drone frame to world frame
        # p_camera_world = p_drone_world + R_world_to_drone * p_camera_drone
        camera_position_world = drone_position + R_world_to_drone.apply(self.camera_position_in_drone)

        # Combine rotations: R_world_to_camera = R_world_to_drone * R_drone_to_camera
        R_world_to_camera = R_world_to_drone * self.R_drone_to_camera
        camera_orientation_quat = R_world_to_camera.as_quat()  # [x, y, z, w]

        return camera_position_world, camera_orientation_quat

    def publish_world_to_camera_transform(self, drone_position: np.ndarray, drone_orientation_quat: np.ndarray):
        """
        Publish the transform from world to camera.

        Args:
            drone_position: [x, y, z] position of drone in world frame
            drone_orientation_quat: [x, y, z, w] quaternion of drone orientation in world frame
        """
        # Compute combined transform
        camera_pos, camera_quat = self.compute_world_to_camera_transform(
            drone_position, drone_orientation_quat
        )

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.world_frame_id
        t.child_frame_id = self.camera_frame_id

        t.transform.translation.x = float(camera_pos[0])
        t.transform.translation.y = float(camera_pos[1])
        t.transform.translation.z = float(camera_pos[2])

        t.transform.rotation.x = float(camera_quat[0])
        t.transform.rotation.y = float(camera_quat[1])
        t.transform.rotation.z = float(camera_quat[2])
        t.transform.rotation.w = float(camera_quat[3])

        self.tf_broadcaster.sendTransform(t)

    def cleanup(self):
        """Cleanup the node."""
        self.destroy_node()

