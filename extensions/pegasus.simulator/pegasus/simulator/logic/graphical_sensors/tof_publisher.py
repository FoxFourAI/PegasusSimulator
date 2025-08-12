#!/usr/bin/env python
"""
| File: tof_publisher.py
| Author: Nataliia Yurevych (nataliia.yurevych@foxfour.ai)
| Description: This file implements a ToFPublisher class that publish ToF sensor data to specific topics.
"""

__all__ = ["ToFPublisher"]

from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from omni.isaac.sensor import Camera

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation
import numpy as np

# ROS2 imports for custom publishing
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
    from geometry_msgs.msg import Point32
    import struct
    ROS2_AVAILABLE = True
except ImportError:
    print("ROS2 not available. ToF data will only be available in debug output")
    ROS2_AVAILABLE = False

class ToFPublisher:
    """
    This class implements a custom ROS2 publisher for ToF sensor data.
    """
    def __init__(self, node_name="tof_publisher"):
        if not ROS2_AVAILABLE:
            return

        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init()

        # Create ROS2 node
        self.node = rclpy.create_node(node_name)

        # Create publishers for each ToF sensor
        self.publishers = {}

    def create_tof_publisher(self, sensor_name, namespace="drone"):
        """Create a ROS2 publisher for a specific ToF sensor"""
        if not ROS2_AVAILABLE:
            return

        topic_name = f"/{namespace}/sensors/{sensor_name}"

        # Publisher for distance grid
        grid_publisher = self.node.create_publisher(
            Float32MultiArray,
            f"{topic_name}/distance_grid",
            10
        )

        # Publisher for closest distance
        closest_publisher = self.node.create_publisher(
            Float32MultiArray,
            f"{topic_name}/closest_distances",
            10
        )

        self.publishers[sensor_name] = {
            'grid': grid_publisher,
            'closest': closest_publisher
        }

        print(f"Created ROS2 publishers for {sensor_name}:")
        print(f"  - {topic_name}/distance_grid")
        print(f"  - {topic_name}/closest_distance")

    def publish_tof_data(self, sensor_name, measurements, closest_distances):
        """Publish ToF sensor data to ROS2"""
        if not ROS2_AVAILABLE or sensor_name not in self.publishers:
            return

        # Publish distance grid
        grid_msg = Float32MultiArray()
        grid_msg.layout.dim.append(MultiArrayDimension())
        grid_msg.layout.dim.append(MultiArrayDimension())
        grid_msg.layout.dim[0].label = "height"
        grid_msg.layout.dim[0].size = 4 #! TODO: Replace with constants
        grid_msg.layout.dim[0].stride = 16
        grid_msg.layout.dim[1].label = "width"
        grid_msg.layout.dim[1].size = 4
        grid_msg.layout.dim[1].stride = 4
        grid_msg.layout.data_offset = 0
        grid_msg.data = measurements.flatten().tolist()

        self.publishers[sensor_name]['grid'].publish(grid_msg)

        # Publish the closest distance
        closest_msg = Float32MultiArray()
        grid_msg.layout.dim.append(MultiArrayDimension())
        grid_msg.layout.dim[0].label = "width"
        grid_msg.layout.dim[0].size = 4 #! TODO: Replace with constants
        grid_msg.layout.dim[0].stride = 4
        grid_msg.layout.data_offset = 0
        closest_msg.data = closest_distances

        self.publishers[sensor_name]['closest'].publish(closest_msg)

    def spin_once(self):
        """Process ROS2 callbacks"""
        if ROS2_AVAILABLE:
            rclpy.spin_once(self.node, timeout_sec=0.0)
