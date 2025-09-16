from pegasus.simulator.logic.graphical_sensors.lidar_utils.configuration_utils import extract_xyz_from_annotator
from collections import defaultdict
import numpy as np

def create_obstacle_distances_from_tof_sensors(lidar_configs):
    """Create obstacle distances array directly from ToF sensor data"""
    # Initialize with max values (no obstacle detected)
    obstacle_distances = [65535] * 72  # 72 sectors at 5° each = 360°

    for config in lidar_configs:
        sensor = config["sensor"]
        sensor_prim = sensor.prim

        # Get closest measurements (4 values, one per column)
        points = extract_xyz_from_annotator(config["pc_annotator"])
        if points is not None:
            closest_measurements = get_closest_measurements(sensor_prim, points) # Closest distances out of columns
        else:
            num_cols = sensor_prim.GetAttribute("omni:sensor:Core:numLines").Get()
            closest_measurements = [65535] * num_cols

        # Convert from meters to centimeters
        closest_distances_cm = [
            int(distance_m * 100) if int(distance_m * 100) < 65535 else 65535
            for distance_m in closest_measurements
        ]

        center_sector = config["sector"]
        half_sectors = 2  # 2 sectors on each side (one LiDAR covers 5 sectors out of 72)

        # Update the center sector and adjacent sectors
        column = 0
        for i in range(0, half_sectors*2):
            sector_idx = (center_sector + i) % 72

            if i == 2:
                # Center sector: use average of adjacent measurements
                if closest_measurements[1] < 65535 and closest_measurements[3] < 65535:
                    obstacle_distances[sector_idx] = int((closest_distances_cm[1] + closest_distances_cm[3]) / 2)
                else:
                    obstacle_distances[sector_idx] = min(closest_distances_cm[1], closest_distances_cm[3])
            else:
                # Side sectors: use individual measurements
                obstacle_distances[sector_idx] = closest_distances_cm[column]
                column += 1

    return obstacle_distances

def get_closest_measurements(sensor_prim, points):
    num_cols = sensor_prim.GetAttribute("omni:sensor:Core:numLines").Get()
    num_rows = sensor_prim.GetAttribute("omni:sensor:Core:numRaysPerLine").Get()[0] # Has num_cols values

    closest_measurements = [0 for _ in range(num_cols)]

    sorted_indices = np.argsort(points[:, 1])
    sorted_points = points[sorted_indices]

    for col_idx in range(num_cols):
        start_idx = col_idx * num_rows
        end_idx = start_idx + num_rows

        column_points = sorted_points[start_idx:end_idx]
        sorted_column_indices = np.argsort(column_points[:, 0]) # Sort by X
        sorted_column_points = column_points[sorted_column_indices]
        min_distance = 65535
        for i, point in enumerate(sorted_column_points):
            # Euclidian distance from origin
            distance = np.sqrt(np.sum(point**2))
            min_distance = min(min_distance, distance)

        closest_measurements[col_idx] = min_distance

    return closest_measurements


    # def publish_lidar_data_to_mavlink(self):
    #     """Publish LiDAR sensor data to MAVLink using obstacle distance"""
    #     if self.ardupilot_backend is None:
    #         return
    #
    #     # Initialize obstacle distances array (72 sectors at 5° each = 360°)
    #     obstacle_distances = [65535] * 72
    #
    #     # Map the 6 LiDAR sensors to sectors
    #     sensor_sector_map = {
    #         'lidar_front': 0,          # 0° forward
    #         'lidar_left_front': 12,    # 60° (12 * 5° = 60°)
    #         'lidar_left_back': 24,     # 120° (24 * 5° = 120°)
    #         'lidar_back': 36,          # 180° (36 * 5° = 180°)
    #         'lidar_right_back': 48,    # 240° (48 * 5° = 240°)
    #         'lidar_right_front': 60,   # 300° (60 * 5° = 300°)
    #     }
    #
    #     for sensor_name, annotator in self.lidar_annotators.items():
    #         if sensor_name not in sensor_sector_map:
    #             continue
    #
    #         try:
    #             data = annotator.get_data()
    #             if data is not None and 'data' in data:
    #                 points = data['data']
    #
    #                 if len(points) > 0:
    #                     # Find closest distance in the point cloud
    #                     distances = []
    #                     for point in points:
    #                         distance = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    #                         distances.append(distance)
    #
    #                     if distances:
    #                         # Convert to centimeters and clamp to valid range
    #                         closest_distance_cm = int(min(distances) * 100)
    #                         closest_distance_cm = max(5, min(65535, closest_distance_cm))
    #
    #                         # Update obstacle distance array
    #                         center_sector = sensor_sector_map[sensor_name]
    #                         obstacle_distances[center_sector] = closest_distance_cm
    #
    #         except Exception as e:
    #             print(f"Error processing LiDAR data for MAVLink {sensor_name}: {e}")
    #
    #     # Send obstacle distances to ArduPilot
    #     try:
    #         self.ardupilot_backend._sensor_data.obstacle_distances = obstacle_distances
    #         self.ardupilot_backend._sensor_data.new_obstacle_data = True
    #     except Exception as e:
    #         print(f"Error sending obstacle data to MAVLink: {e}")
