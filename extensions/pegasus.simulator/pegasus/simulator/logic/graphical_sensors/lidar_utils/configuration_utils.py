""""
| File: configuration_utils.py
| Author: Nataliia Yurevych (nataliia.yurevych@foxfour.ai)
| Description: This file contains functions for LidarRtx configuration.
"""

from pegasus.simulator.logic.graphical_sensors.lidar_utils.custom_lidar_config import LidarConfig

import omni.replicator.core as rep
import numpy as np
from pxr import Gf, Vt


def configure_solid_state_lidar(sensor_obj, config: LidarConfig):
    """
    Complete configuration for solid state LiDAR
    """
    sensor_prim = sensor_obj.prim

    total_scan_points = configure_emitter_elevations(sensor_obj, config)

    current_ranges_min = sensor_prim.GetAttribute("omni:sensor:Core:rangesMinM").Get()
    ranges_min_array = [config.min_range] * len(current_ranges_min)
    ranges_min_vt = Vt.FloatArray(ranges_min_array)

    current_ranges_max = sensor_prim.GetAttribute("omni:sensor:Core:rangesMaxM").Get()
    ranges_max_array = [config.max_range] * len(current_ranges_max)
    ranges_max_vt = Vt.FloatArray(ranges_max_array)

    rays_per_line_array = [config.h_res] * config.v_res
    rays_per_line_vt = Vt.UIntArray(rays_per_line_array)

    attributes = {
        "omni:sensor:Core:numLines": config.v_res,
        "omni:sensor:Core:numRaysPerLine": rays_per_line_vt,
        "omni:sensor:Core:scanType": "SOLID_STATE",
        "omni:sensor:Core:nearRangeM": config.min_range,
        "omni:sensor:Core:farRangeM": config.max_range,
        "omni:sensor:Core:rangeCount": 1,
        "omni:sensor:Core:rangesMinM": Vt.FloatArray([config.min_range]),
        "omni:sensor:Core:rangesMaxM": Vt.FloatArray([config.max_range]),
        "omni:sensor:Core:rangesMinM": ranges_min_vt,
        "omni:sensor:Core:rangesMaxM": ranges_max_vt,
        "omni:sensor:Core:scanRateBaseHz": config.scan_rate,
        "omni:sensor:Core:reportRateBaseHz": config.scan_rate,
        "omni:sensor:Core:numberOfEmitters": total_scan_points,
        "omni:sensor:Core:numberOfChannels": total_scan_points,
        "omni:sensor:Core:rangeResolutionM": config.range_resolution,
        "omni:sensor:Core:rangeAccuracyM": config.range_accuracy,
        "omni:sensor:Core:maxReturns": config.max_returns,
        "omni:sensor:Core:minReflectance": 0.0,
        "omni:sensor:Core:avgPowerW": config.avg_power_w,
        "omni:sensor:Core:waveLengthNm": config.wave_length_nm,
        "omni:sensor:Core:pulseTimeNs": config.pulse_time_ns,
        "omni:sensor:Core:azimuthErrorStd": config.azimuth_error_std,
        "omni:sensor:Core:elevationErrorStd": config.elevation_error_std,
    }

    for param_name, param_value in attributes.items():
        sensor_prim.GetAttribute(param_name).Set(param_value)

    return total_scan_points

def configure_emitter_elevations(sensor_obj, config: LidarConfig):
    """
    Configure complete emitter states with azimuth and elevation arrays for solid state LiDAR
    """
    sensor_prim = sensor_obj.prim

    h_start, h_end = config.h_range
    v_start, v_end = config.v_range

    # Generate azimuth angles (horizontal sweep)
    azimuth_angles = np.linspace(h_start, h_end, config.h_res, endpoint=False)
    for idx, angle in enumerate(azimuth_angles):
        azimuth_angles[idx] = (angle + h_end/4 + 360) % 360
        print(f"Azimuth angle {angle} became {azimuth_angles[idx]}")

    # Generate elevation angles (vertical levels)
    elevation_angles = np.linspace(v_start, v_end, config.v_res, endpoint=False)
    for idx, angle in enumerate(elevation_angles):
        elevation_angles[idx] = angle + v_end/4
        print(f"Elevation angle {angle} became {elevation_angles[idx]}")

    # Create full scan pattern arrays
    full_azimuth_array = []
    full_elevation_array = []

    for elev in elevation_angles:
        for azim in azimuth_angles:
            full_azimuth_array.append(azim)
            full_elevation_array.append(elev)

    # Convert to USD arrays
    azimuth_vt = Vt.FloatArray(full_azimuth_array)
    elevation_vt = Vt.FloatArray(full_elevation_array)

    print(f"Full azimuth: {full_azimuth_array}, Full elevation: {full_elevation_array}")
    print(f"Full azimuth_vt: {azimuth_vt}, Full elevation_vt: {elevation_vt}")

    # Configure emitter state s001 (usually the main/only emitter state)
    emitter_id = "s001"

    # Set azimuth array
    azimuth_attr = f"omni:sensor:Core:emitterState:{emitter_id}:azimuthDeg"
    sensor_prim.GetAttribute(azimuth_attr).Set(azimuth_vt)

    # Set elevation array
    elevation_attr = f"omni:sensor:Core:emitterState:{emitter_id}:elevationDeg"
    sensor_prim.GetAttribute(elevation_attr).Set(elevation_vt)

    # Create arrays for emitter state parameters
    bank_array = [i for i in range(config.h_res) for _ in range(config.v_res)]

    total_points = len(full_azimuth_array)

    channel_ids = list(range(1, total_points + 1))  # Sequential channel IDs
    fire_times = [i * 82500 for i in range(total_points)]  # Fire times in nanoseconds
    range_ids = [0] * total_points  # All use range 0

    def as_vt_uint(seq):
        if isinstance(seq, np.ndarray):
            return Vt.UIntArray(seq.astype(np.uint32).tolist())
        return Vt.UIntArray([int(x) for x in seq])

    def as_vt_float(seq):
        if isinstance(seq, np.ndarray):
            return Vt.FloatArray(seq.astype(np.float32).tolist())
        return Vt.FloatArray([float(x) for x in seq])

    fire_times_u32 = np.asarray(fire_times, dtype=np.uint32)
    emitter_array_params = {
        f"omni:sensor:Core:emitterState:{emitter_id}:bank": as_vt_uint(bank_array),
        f"omni:sensor:Core:emitterState:{emitter_id}:channelId": as_vt_uint(channel_ids),
        f"omni:sensor:Core:emitterState:{emitter_id}:fireTimeNs": as_vt_uint(fire_times_u32),
        f"omni:sensor:Core:emitterState:{emitter_id}:rangeId": as_vt_uint(range_ids),
    }

    for param_name, param_array in emitter_array_params.items():
        sensor_prim.GetAttribute(param_name).Set(param_array)

    # Set float arrays for emitter parameters
    zeros = [0.0] * total_points
    distancesM = [0.05] * total_points
    horOffsets = [0.0] * total_points
    reportRates = [1.0] * total_points

    attributes = {
        f"omni:sensor:Core:emitterState:{emitter_id}:distanceCorrectionM": as_vt_float(zeros),
        f"omni:sensor:Core:emitterState:{emitter_id}:focalDistM": as_vt_float(distancesM),
        f"omni:sensor:Core:emitterState:{emitter_id}:focalSlope": as_vt_float(zeros),
        f"omni:sensor:Core:emitterState:{emitter_id}:horOffsetM": as_vt_float(horOffsets),
        f"omni:sensor:Core:emitterState:{emitter_id}:vertOffsetM": as_vt_float(horOffsets),
        f"omni:sensor:Core:emitterState:{emitter_id}:reportRateDiv": as_vt_float(reportRates),
        f"omni:sensor:Core:emitterState:{emitter_id}:isROIState": False
    }

    for attr, value in attributes.items():
        sensor_prim.GetAttribute(attr).Set(value)

    return total_points

def create_lidar_annotators(render_product):
    """
    Attach a point-cloud annotator to the LiDAR's render product.
    Returns an annotator.
    """

    ann = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
    ann.attach([render_product])  # attach expects a list
    print(f"[Annotator] Attached IsaacExtractRTXSensorPointCloudNoAccumulator to render product: {render_product}")

    return ann

def attach_pc_annotator(render_product):
    # Prefer per-frame extractor; falls back to other known names if needed
    ann = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
    ann.attach([getattr(render_product, "path", render_product)])
    print(f"[Annotator] Attached IsaacExtractRTXSensorPointCloudNoAccumulator")

    return ann

def extract_xyz_from_annotator(ann):
    """Return Nx3 float32 or None if not ready yet."""
    if ann is None:
        return None

    out = ann.get_data()

    if out is None:
        return None

    if "data" in out:
        arr = np.asarray(out["data"], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]
    return


def print_verification(sensor):
    """Print final sensor configuration for verification."""
    print(f"\n=== Final Sensor Configuration ===")
    print(f"Prim path: {sensor.prim_path}")
    print(f"Render product: {sensor.get_render_product_path()}")

    # Verify final configuration
    sensor_prim = sensor.prim
    verification_attrs = [
        "omni:sensor:Core:validStartAzimuthDeg",
        "omni:sensor:Core:validEndAzimuthDeg",
        "omni:sensor:Core:numLines",
        "omni:sensor:Core:scanType",
        "omni:sensor:Core:nearRangeM",
        "omni:sensor:Core:farRangeM",
        "omni:sensor:Core:scanRateBaseHz"
    ]

    print(f"\nVerification - Final values:")
    for attr_name in verification_attrs:
        value = sensor_prim.GetAttribute(attr_name).Get()
        print(f"  {attr_name}: {value}")

    # Verify emitter state arrays
    print(f"\nEmitter State Arrays:")
    emitter_id = "s001"

    array_attrs = ["azimuthDeg", "elevationDeg", "bank", "channelId", "fireTimeNs", "rangeId"]
    for array_type in array_attrs:
        attr_name = f"omni:sensor:Core:emitterState:{emitter_id}:{array_type}"
        array_data = sensor_prim.GetAttribute(attr_name).Get()
        if array_data and len(array_data) > 0:
            print(f"  {array_type}: {len(array_data)} elements")
            if array_type in ["azimuthDeg", "elevationDeg"]:
                print(f"    Range: {min(array_data):.1f}° to {max(array_data):.1f}°")
            elif array_type == "fireTimeNs" and len(array_data) > 1:
                print(f"    Time step: {array_data[1] - array_data[0]} ns")
            elif array_type == "channelId" and len(array_data) > 1:
                print(f"    Channel range: {min(array_data)} to {max(array_data)}")
        else:
            print(f"  {array_type}: empty")

    # Verify numRaysPerLine array
    numrays_attr = "omni:sensor:Core:numRaysPerLine"
    rays_array = sensor_prim.GetAttribute(numrays_attr).Get()
    if rays_array:
        print(f"  numRaysPerLine: {len(rays_array)} lines, {rays_array[0]} rays each")
    else:
        print(f"  numRaysPerLine: empty")
