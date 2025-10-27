from pegasus.simulator.logic.graphical_sensors.lidar_utils.configuration_utils import extract_xyz_from_annotator
from collections import defaultdict
import numpy as np
import line_profiler

# @line_profiler.profile
def create_obstacle_distances_from_tof_sensors(lidar_configs):
    # return [65535] * 72
    # Initialize with max values (no obstacle detected)
    obstacle_distances = [65535] * 72  # 72 sectors at 5 degrees each = 360 degrees

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

        closest_distances_cm = closest_distances_cm[::-1]

        sectors = config["sector"]
        # for idx, sector in enumerate(sectors):
        #     if idx == 2:
        #         obstacle_distances[sector % 72] = int((closest_distances_cm[1] + closest_distances_cm[2]) / 2)
        #         continue
        #     if idx > 2:
        #         obstacle_distances[sector % 72] = closest_distances_cm[idx-1]
        #     else:
        #         obstacle_distances[sector % 72] = closest_distances_cm[idx]

        for idx, sector in enumerate(sectors):
            obstacle_distances[sector % 72] = closest_distances_cm[idx]

    return obstacle_distances

def normalize_deg(a):
    """Map degrees to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    # Put +180 into -180 bin for consistent half-open edges later
    a[a == 180.0] = -180.0
    return a

def bin_points_by_nearest_azimuth(points_xyz, centers):
    """
    Group raw LiDAR hits into columns by nearest azimuth center (degrees).
    Returns: list of length M (num_cols); each item is array of points.
    """
    if points_xyz.size == 0:
        return [np.empty((0, 3), dtype=np.float32) for _ in range(len(col_centers_deg))]

    # angles of points (sensor frame!)
    theta = np.rad2deg(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]))  # atan2(y, x)
    theta = normalize_deg(theta)

    # Broadcast and pick nearest center using circular difference
    # diff[i, j] = circular distance between theta[i] and centers[j]
    diff = np.abs(normalize_deg(theta[:, None] - centers[None, :]))  # (N, M)
    idx = np.argmin(diff, axis=1)  # column index for each point

    M = len(centers)
    bins = [np.empty((0, 3), dtype=points_xyz.dtype) for _ in range(M)]
    for j in range(M):
        sel = (idx == j)
        if np.any(sel):
            bins[j] = points_xyz[sel]
    return bins

def closest_per_column_nearest(points_xyz, col_centers_deg, max_range=np.inf):
    """
    Returns:
      distances: (M,) min Euclidean distance per column (or max_range if none)
      points:    list length M; the (3,) point that achieved the min (or None)
    """
    cols = bin_points_by_nearest_azimuth(points_xyz, col_centers_deg)
    M = len(cols)
    d_min = np.full(M, max_range, dtype=np.float64)
    p_min = [None] * M
    for j, arr in enumerate(cols):
        if arr.shape[0]:
            d = np.linalg.norm(arr, axis=1)
            k = int(np.argmin(d))
            d_min[j] = d[k]
            p_min[j] = arr[k]
    return d_min, p_min


def get_closest_measurements(sensor_prim, points):
    num_cols = sensor_prim.GetAttribute("omni:sensor:Core:numLines").Get()
    num_rows = sensor_prim.GetAttribute("omni:sensor:Core:numRaysPerLine").Get()[0] # Has num_cols values

    emitter_id = "s001"

    # Set azimuth array
    azimuth_attr = f"omni:sensor:Core:emitterState:{emitter_id}:azimuthDeg"
    az = sensor_prim.GetAttribute(azimuth_attr).Get()

    az = np.asarray(az, dtype=np.float64)

    col_centers = az[:num_cols] # Take the first num_cols elements which represent each column
    col_centers = normalize_deg(col_centers)

    closest_dists, closest_points = closest_per_column_nearest(points, col_centers, max_range=65535.0)


    return closest_dists

def send_lidar_data_to_mavlink(lidar_configs, ardupilot_backend):
    if ardupilot_backend is None:
        print("No ArduPilot backend! Cannot send LiDARs data to MAVLink!")
        return

    # Create obstacle distances array (72 sectors at 5 degrees each = 360 degrees)
    obstacle_distances = create_obstacle_distances_from_tof_sensors(lidar_configs)

    # Send obstacle distances to ArduPilot
    ardupilot_backend._sensor_data.obstacle_distances = obstacle_distances
    ardupilot_backend._sensor_data.new_obstacle_data = True
