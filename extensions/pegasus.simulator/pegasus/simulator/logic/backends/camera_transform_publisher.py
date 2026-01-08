class CameraDroneTransformPublisher(Node):
    """
    ROS2 Node that publishes the static transform between camera and drone body frame,
    as well as dynamic transforms for the drone in the world.
    """

    def __init__(self, camera_config: dict, drone_frame_id: str = "drone", camera_frame_id: str = "camera"):
        super().__init__('camera_drone_tf_publisher')

        self.drone_frame_id = drone_frame_id
        self.camera_frame_id = camera_frame_id

        # Static transform broadcaster for camera-to-drone (doesn't change)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Dynamic transform broadcaster for drone-to-world
        self.dynamic_tf_broadcaster = TransformBroadcaster(self)

        # Extract camera pose relative to drone body from config
        self.camera_position = camera_config.get("position", np.array([0.0, 0.0, 0.0]))
        self.camera_orientation_euler = camera_config.get("orientation", np.array([0.0, 0.0, 0.0]))

        # Convert euler angles (degrees) to quaternion
        # The orientation is given as [roll, pitch, yaw] in degrees
        rotation = Rotation.from_euler('XYZ', self.camera_orientation_euler, degrees=True)
        self.camera_orientation_quat = rotation.as_quat()  # [x, y, z, w]

        # Publish the static transform once
        self._publish_static_camera_transform()

        self.get_logger().info(
            f"Camera-Drone TF Publisher initialized:\n"
            f"  Camera position (in drone frame): {self.camera_position}\n"
            f"  Camera orientation (euler deg): {self.camera_orientation_euler}\n"
            f"  Camera orientation (quat xyzw): {self.camera_orientation_quat}"
        )

    def _publish_static_camera_transform(self):
        """Publish the static transform from drone body to camera."""
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.drone_frame_id  # Parent frame (drone body)
        t.child_frame_id = self.camera_frame_id   # Child frame (camera)

        # Translation (camera position relative to drone body)
        t.transform.translation.x = float(self.camera_position[0])
        t.transform.translation.y = float(self.camera_position[1])
        t.transform.translation.z = float(self.camera_position[2])

        # Rotation (camera orientation relative to drone body)
        # scipy returns [x, y, z, w] format
        t.transform.rotation.x = float(self.camera_orientation_quat[0])
        t.transform.rotation.y = float(self.camera_orientation_quat[1])
        t.transform.rotation.z = float(self.camera_orientation_quat[2])
        t.transform.rotation.w = float(self.camera_orientation_quat[3])

        self.static_tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Published static transform: {self.drone_frame_id} -> {self.camera_frame_id}")

    def publish_drone_world_transform(self, position: np.ndarray, orientation_quat: np.ndarray):
        """
        Publish the dynamic transform from world to drone.

        Args:
            position: [x, y, z] position of drone in world frame
            orientation_quat: [x, y, z, w] quaternion of drone orientation in world frame
        """
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = self.drone_frame_id

        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])

        t.transform.rotation.x = float(orientation_quat[0])
        t.transform.rotation.y = float(orientation_quat[1])
        t.transform.rotation.z = float(orientation_quat[2])
        t.transform.rotation.w = float(orientation_quat[3])

        self.dynamic_tf_broadcaster.sendTransform(t)

    def cleanup(self):
        """Cleanup the node."""
        self.destroy_node()

