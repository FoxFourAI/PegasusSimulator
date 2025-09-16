""""
| File: lidar_nodes.py
| Author: Nataliia Yurevych (nataliia.yurevych@foxfour.ai)
| Description: This file contains ROS2 and RTXLidar related functions.
"""

import numpy as np
from math import sin, cos, radians

# ROS2
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
    from geometry_msgs.msg import Point32
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2, PointField
    from geometry_msgs.msg import TransformStamped
    import struct
    ROS2_AVAILABLE = True
except ImportError as e:
    print(f"ROS2 is not available in the lidar_nodes.py file.\n{e}")
    ROS2_AVAILABLE = False

def make_pointcloud2_xyz32(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    Pack Nx3 float32 XYZ into a sensor_msgs/PointCloud2.
    - points_xyz: shape (N, 3), dtype float32/float64
    - frame_id: e.g., "lidar_link"
    - stamp: rclpy time -> header.stamp (node.get_clock().now().to_msg())
    """
    if points_xyz is None:
        return None
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width  = pts.shape[0]

    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12  # 3 * 4B
    msg.row_step   = msg.point_step * msg.width
    msg.is_dense   = True

    # Pack as contiguous bytes
    msg.data = pts[:, :3].astype(np.float32, copy=False).tobytes()
    return msg

def make_pointcloud2_xyz32i(points_xyz: np.ndarray, intensities: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    Variant with intensity (Nx1). Offsets: x=0,y=4,z=8,intensity=12 (point_step=16).
    """
    if points_xyz is None or intensities is None:
        return None
    pts = np.asarray(points_xyz, dtype=np.float32)
    inten = np.asarray(intensities, dtype=np.float32).reshape(-1, 1)
    if pts.shape[0] != inten.shape[0]:
        return None
    packed = np.concatenate([pts[:, :3], inten], axis=1)  # (N,4)

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width  = packed.shape[0]

    msg.fields = [
        PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16  # 4 * 4B
    msg.row_step   = msg.point_step * msg.width
    msg.is_dense   = True
    msg.data = packed.astype(np.float32, copy=False).tobytes()
    return msg

class Ros2PointCloudPublisher(Node):
    def __init__(self, topic="lidar/points", frame_id="lidar_link"):
        super().__init__("isaac_lidar_pub")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.pub = self.create_publisher(PointCloud2, topic, qos)
        self.frame_id = frame_id

    def publish_points(self, points_xyz: np.ndarray):
        if points_xyz is None or points_xyz.size == 0:
            return
        stamp = self.get_clock().now().to_msg()
        msg = make_pointcloud2_xyz32(points_xyz, self.frame_id, stamp)
        if msg is not None:
            self.pub.publish(msg)
