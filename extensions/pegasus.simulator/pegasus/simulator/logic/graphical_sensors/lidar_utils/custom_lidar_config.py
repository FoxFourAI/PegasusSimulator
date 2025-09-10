import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import carb
import omni
from isaacsim.core.api import SimulationContext
from isaacsim.sensors.rtx import LidarRtx
from isaacsim.core.utils import stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Vt

@dataclass
class LidarConfig:
    """Configuration parameters for the LiDAR sensor."""
    # config_name: str = "Example_Solid_State"
    config_name: str = None
    h_fov: float = 22.0              # Horizontal FOV in degrees
    v_fov: float = 16.0              # Vertical FOV in degrees
    h_res: int = 4                   # Horizontal resolution
    v_res: int = 4                   # Vertical resolution
    min_range: float = 0.5           # Minimum range in meters
    max_range: float = 100.0         # Maximum range in meters
    scan_rate: float = 10.0          # Scan rate in Hz

    # Quality parameters
    range_resolution: float = 0.004
    range_accuracy: float = 0.025
    # max_returns: int = 2
    max_returns: int = 1
    min_reflectance: float = 0.1
    avg_power_w: float = 0.06
    wave_length_nm: float = 1550.0
    pulse_time_ns: float = 6
    azimuth_error_std: float = 0.025
    elevation_error_std: float = 0.025

    # Render product
    hydra_texture = None

    @property
    def total_scan_points(self) -> int:
        """Total number of scan points."""
        return self.h_res * self.v_res

    @property
    def h_range(self) -> Tuple[float, float]:
        """Horizontal FOV range as (start, end) in degrees."""
        return (-self.h_fov/2, self.h_fov/2)
        # return (0, self.h_fov)

    @property
    def v_range(self) -> Tuple[float, float]:
        """Vertical FOV range as (start, end) in degrees."""
        return (-self.v_fov/2, self.v_fov/2)
        # return (0, self.v_fov)
