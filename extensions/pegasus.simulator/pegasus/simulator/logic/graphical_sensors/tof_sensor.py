"""
| File: tof_sensor.py
| Author: Nataliia Yurevych (nataliia.yurevych@foxfour.ai)
| Description: Simulates a Time-of-Flight (ToF) sensor with 4x4 resolution and configurable FoV.
"""
__all__ = ["ToFSensor"]

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.graphical_sensors import GraphicalSensor
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from omni.isaac.sensor import Camera
from omni.usd import get_stage_next_free_path

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation

class ToFSensor(GraphicalSensor):
    """
    The class that implements a Time-of-Flight sensor. This class inherits the base class GraphicalSensor because
    in simulation we can't actually emit light rays, so we simulate this by rendering depth information.
    The sensor provides a configurable grid of distance measurements with configurable field of view.
    """

    def __init__(self, sensor_name, config={}):
        """
        Initialize the ToFSensor class

        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the ToFSensor - it can be empty or only have some of the parameters used by the ToFSensor.

        Examples:
            The dictionary default parameters are

            >>> {"position": np.array([0.10, 0.0, 0.0]),
            >>> "orientation": np.array([0.0, 0.0, 0.0]),
            >>> "horizontal_fov": 22.0,  # degrees
            >>> "vertical_fov": 16.0,    # degrees
            >>> "frequency": 60.0,       # Hz # TODO: Replace with 50 Hz (it breaks the simulation due to rendering frequency must be divisible by camera frequency)
            >>> "max_range": 5.0,        # meters # TODO: Replace with real data
            >>> "min_range": 0.05,       # meters # TODO: Replace with real data
            >>> "resolution": (4, 4),    # 4x4 grid
            >>> "noise_std": 0.01}       # standard deviation of noise in meters
        """

        # Initialize the Super class "object" attributes
        super().__init__(sensor_type="ToFSensor", update_rate=config.get("frequency", 50.0))

        # Setup the name of the sensor primitive path
        self._sensor_name = sensor_name
        self._stage_prim_path = ""

        # Configurations of the ToF sensor
        self._position = config.get("position", np.array([0.10, 0.0, 0.0]))
        self._orientation = config.get("orientation", np.array([0.0, 0.0, 0.0]))
        self._horizontal_fov = config.get("horizontal_fov", 22.0)
        self._vertical_fov = config.get("vertical_fov", 16.0)
        self._frequency = config.get("frequency", 60.0)
        self._max_range = config.get("max_range", 5.0)
        self._min_range = config.get("min_range", 0.05)
        self._resolution = config.get("resolution", (4, 4))
        self._noise_std = config.get("noise_std", 0.01)

        # Internal camera resolution
        self._camera_resolution = (640, 480)

        # An empty sensor output dictionary
        self._state = {}
        self._sensor_full_set = False

        # Initialize the camera object (will be created in initialize method)
        self._camera = None

        self.counter = 0

    def initialize(self, vehicle):
        """Initialize the ToF sensor with the vehicle"""

        # Initialize the Super class "object" attributes
        super().initialize(vehicle)

        # Get the complete stage prefix for the sensor
        self._stage_prim_path = get_stage_next_free_path(
            PegasusInterface().world.stage,
            self._vehicle.prim_path + "/body/" + self._sensor_name,
            False
        )

        # Get the sensor name that was actually created and update the sensor name
        self._sensor_name = self._stage_prim_path.rpartition("/")[-1]

        # Create the camera object attached to the rigid body vehicle (used for depth sensing)
        self._camera = Camera(
            prim_path=self._stage_prim_path,
            frequency=self._frequency,
            resolution=self._camera_resolution
        )

        # Set the camera position locally with respect to the drone
        self._camera.set_local_pose(
            np.array(self._position),
            Rotation.from_euler("ZYX", self._orientation, degrees=True).as_quat()
        )

    def start(self):
        """Start the ToF sensor"""

        # Start the camera
        self._camera.initialize()

        # Set the camera to capture depth information
        self._camera.add_distance_to_image_plane_to_frame()

        # Set the field of view based on the horizontal FoV
        # Isaac Sim uses horizontal FoV for camera configuration
        self._camera.set_horizontal_aperture(self._horizontal_fov)

        # Set clipping range
        self._camera.set_clipping_range(self._min_range, self._max_range)

        # Signal that the sensor is fully set
        self._sensor_full_set = True

    def stop(self):
        """Stop the ToF sensor"""
        self._sensor_full_set = False

    @property
    def state(self):
        """
        Return the data produced by the sensor
        """
        return self._state

    def _process_depth_image(self, depth_image):
        """
        Process the depth image to extract 4x4 ToF measurements

        Args:
            depth_image (np.ndarray): Full resolution depth image

        Returns:
            np.ndarray: 4x4 array of distance measurements
        """
        if depth_image is None:
            return np.full(self._resolution, self._max_range)

        height, width = depth_image.shape
        grid_height, grid_width = self._resolution[1], self._resolution[0]

        # Calculate the size of each zone
        zone_height = height // grid_height
        zone_width = width // grid_width

        # Initialize the output grid
        tof_measurements = np.zeros((grid_height, grid_width))

        # Process each zone in the 4x4 grid
        for i in range(grid_height):
            for j in range(grid_width):
                # Calculate zone boundaries
                y_start = i * zone_height
                y_end = min((i + 1) * zone_height, height)
                x_start = j * zone_width
                x_end = min((j + 1) * zone_width, width)

                # Extract the zone
                zone = depth_image[y_start:y_end, x_start:x_end]

                # Take the minimum distance in the zone (closest object) to detect obstacles
                valid_depths = zone[(zone >= self._min_range) & (zone <= self._max_range)]

                if len(valid_depths) > 0:
                    tof_measurements[i, j] = np.min(valid_depths)
                else:
                    tof_measurements[i, j] = self._max_range

                # Add noise to the measurement
                if self._noise_std > 0:
                    noise = np.random.normal(0, self._noise_std)
                    tof_measurements[i, j] = np.clip(
                        tof_measurements[i, j] + noise,
                        self._min_range,
                        self._max_range
                    )

        return tof_measurements

    @GraphicalSensor.update_at_rate # It ensures running at the configured frequency
    def update(self, state: State, dt: float):
        """Method that gets the current depth measurements from the ToF sensor and returns them as a dictionary.

        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).

        Returns:
            (dict) A dictionary containing the current state of the sensor (the data produced by the sensor)
        """

        # Skip first few updates to ensure camera is properly initialized
        while self.counter < 10:
            self.counter += 1
            return

        # If all the sensor properties are not set yet, return None
        if not self._sensor_full_set:
            return None


        # Get depth image from the camera
        depth_image = self._camera.get_depth()

        # Process the depth image to get 4x4 ToF measurements
        tof_grid = self._process_depth_image(depth_image)

        # Create angular information for each measurement
        # Calculate the angular position of each measurement zone
        # It is each column`s center line is at that angle from the center.
        # For horizontal: [-11°, -3.67°, 3.67°, 11°]
        h_angles = np.linspace(-self._horizontal_fov/2, self._horizontal_fov/2, self._resolution[0])
        v_angles = np.linspace(-self._vertical_fov/2, self._vertical_fov/2, self._resolution[1])

        self._state = {
            "sensor_name": self._sensor_name,
            "stage_prim_path": self._stage_prim_path,
            "measurements": tof_grid,  # 4x4 array of distances
            "horizontal_fov": self._horizontal_fov,
            "vertical_fov": self._vertical_fov,
            "resolution": self._resolution,
            "max_range": self._max_range,
            "min_range": self._min_range,
            "frequency": self._frequency,
            "horizontal_angles": h_angles,  # Angular position of each column
            "vertical_angles": v_angles,    # Angular position of each row
            "timestamp": state.position  # Timestamp placeholder
        }

        return self._state

    def get_measurement_at_zone(self, row, col):
        """
        Get the distance measurement at a specific zone in the 4x4 grid

        Args:
            row (int): Row index (0-3)
            col (int): Column index (0-3)

        Returns:
            float: Distance measurement in meters, or None if invalid
        """

        if 0 <= row < self._resolution[1] and 0 <= col < self._resolution[0]:
            return self._state["measurements"][row, col]

        return None

    def get_closest_measurements(self):
        """
        Get the closest distance measurements from columns

        Returns:
            list: List of 4 closest distance measurements from columns in meters
        """
        if self._state is None or "measurements" not in self._state:
            return self._max_range

        measurements = self._state["measurements"]
        closest_measurements = np.zeros(self._resolution[1])

        for i in range(self._resolution[1]): # Column idx
            column = np.zeros(self._resolution[0]) # Save i-th column
            for j in range(self._resolution[0]): # Row idx
                column[j] = measurements[j][i]
            closest_measurements[i] = np.min(column)

        return closest_measurements

    def get_measurements_as_list(self):
        """
        Get all measurements as a flattened list

        Returns:
            list: List of 16 distance measurements (row-major)
        """

        return self._state["measurements"].flatten().tolist()
