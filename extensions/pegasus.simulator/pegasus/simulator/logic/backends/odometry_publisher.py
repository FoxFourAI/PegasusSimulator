import numpy as np
from scipy.spatial.transform import Rotation

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
    from geometry_msgs.msg import Point32
    import struct
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
    ROS2_AVAILABLE = True
except ImportError as e:
    print(f"ROS2 is not available in the main file.\n{e}")
    ROS2_AVAILABLE = False

if ROS2_AVAILABLE:
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3

class OdometryPublisher:
    def __init__(self, topic_name="/drone/odometry", frame_id="odom", child_frame_id="base_link"):
        if not ROS2_AVAILABLE:
            return

        # Ensure rclpy is initialized
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("odometry_publisher_node")
        self.pub = self.node.create_publisher(Odometry, topic_name, 10)
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id

    def publish(self, multirotor):
        if not ROS2_AVAILABLE:
            return

        # 1. Get State from Isaac Sim
        # Note: get_world_pose returns orientation as (w, x, y, z)
        position, orientation_wxyz = multirotor.get_world_pose()
        lin_vel_world = multirotor.get_linear_velocity()
        ang_vel_body = multirotor.get_angular_velocity()

        # 2. Convert Quaternion (WXYZ -> XYZW for ROS)
        w, x, y, z = orientation_wxyz
        orientation_xyzw = np.array([x, y, z, w])

        # 3. Convert Linear Velocity to Body Frame (Standard for Odometry)
        # Create rotation matrix from quaternion
        rot = Rotation.from_quat([x, y, z, w])
        # Rotate world velocity into body frame: v_body = R_inverse * v_world
        lin_vel_body = rot.inv().apply(lin_vel_world)

        # 4. Create Message
        odom = Odometry()
        odom.header.stamp = self.node.get_clock().now().to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id

        # Pose (World Frame)
        odom.pose.pose.position.x = float(position[0])
        odom.pose.pose.position.y = float(position[1])
        odom.pose.pose.position.z = float(position[2])
        odom.pose.pose.orientation.x = float(orientation_xyzw[0])
        odom.pose.pose.orientation.y = float(orientation_xyzw[1])
        odom.pose.pose.orientation.z = float(orientation_xyzw[2])
        odom.pose.pose.orientation.w = float(orientation_xyzw[3])

        # Twist (Body Frame)
        odom.twist.twist.linear.x = float(lin_vel_body[0])
        odom.twist.twist.linear.y = float(lin_vel_body[1])
        odom.twist.twist.linear.z = float(lin_vel_body[2])
        odom.twist.twist.angular.x = float(ang_vel_body[0])
        odom.twist.twist.angular.y = float(ang_vel_body[1])
        odom.twist.twist.angular.z = float(ang_vel_body[2])

        # 5. Publish
        self.pub.publish(odom)

    def cleanup(self):
        if ROS2_AVAILABLE and self.node:
            self.node.destroy_node()