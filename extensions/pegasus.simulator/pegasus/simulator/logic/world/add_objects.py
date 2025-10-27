from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, DynamicSphere
from omni.isaac.core.materials import PhysicsMaterial
from pxr import Usd, UsdGeom, Sdf
from omni.usd import get_context
import omni.isaac.core.utils.stage as stage_utils
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import os

# Isaac Sim RTX LiDAR imports
from pxr import Gf, Vt, Usd, UsdGeom, UsdPhysics, Sdf

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon) # Number of edges
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1): # Go through all edges
        p2x, p2y = polygon[i % n]
        if (y < p1y) != (y < p2y) and (x < ((y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x)):
            inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def spawn_random_walls(areas, num_walls_per_area):
    """
    Spawn walls at random positions and rotations within defined areas.

    Parameters:
        areas : list of lists
            Each area is a list of 4 points, where each point is [x, y, z]
            Example: [[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]]
        num_walls_per_area : int or list
            Number of walls to spawn in each area (single int applies to all areas)

    Returns:
        list : Information about spawned walls (position, rotation, area_index)
    """

    if isinstance(num_walls_per_area, int):
        num_walls_per_area = [num_walls_per_area] * len(areas)

    spawned_walls = []
    wall_counter = 0

    for area_idx, area_points in enumerate(areas):
        # Extract XY coordinates (ignore Z since it's always the same)
        points_2d = np.array([[p[0], p[1]] for p in area_points])

        # Find bounding box for efficient sampling
        min_x, min_y = points_2d.min(axis=0)
        max_x, max_y = points_2d.max(axis=0)

        walls_spawned = 0
        attempts = 0
        max_attempts = num_walls_per_area[area_idx] * 100

        while walls_spawned < num_walls_per_area[area_idx] and attempts < max_attempts:
            attempts += 1

            # Generate random position within bounding box
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)

            # Check if point is inside the polygon
            if point_in_polygon([x, y], points_2d):
                # Generate random rotation around Z axis (in degrees)
                rotation_z = np.random.uniform(0, 360)

                # Create the wall
                position = np.array([x, y, 0.0])

                # Color
                R = np.random.uniform(0, 255)/255
                G = np.random.uniform(0, 255)/255
                B = np.random.uniform(0, 255)/255

                # Store wall info
                wall_info = {
                    'prim_path': f"/World/Wall_{wall_counter}",
                    'name': f"wall_{wall_counter}",
                    'position': position,
                    'rotation_z': rotation_z,
                    'scale': np.array([1, 1, 20]),
                    'color': np.array([R, G, B]),
                    'area_index': area_idx
                }

                spawned_walls.append(wall_info)

                quat = Rotation.from_euler("XYZ", [0, 0, rotation_z], degrees=True).as_quat()

                # Spawn the wall
                FixedCuboid(
                    prim_path=wall_info['prim_path'],
                    name=wall_info['name'],
                    position=wall_info['position'],
                    scale=wall_info['scale'],
                    color=wall_info['color'],
                    orientation = np.array([quat[3], quat[0], quat[1], quat[2]]) # [w, x, y, z]
                )

                wall_counter += 1
                walls_spawned += 1

        if walls_spawned < num_walls_per_area[area_idx]:
            print(f"Warning: Only spawned {walls_spawned}/{num_walls_per_area[area_idx]} walls in area {area_idx}")

    return spawned_walls

def add_house_numbers():
    """
    Add numbers to some buildings.

    :return: None
    """
    houses_numbers_data = {}
    scale = Gf.Vec3d(1.0, 6.0, 6.0)

    # 897
    houses_numbers_data["897"] = {
        "number_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/duplicated_03/duplicated_03",
        "frame_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/Kasa_Residential_01_low/Kasa_Residential_01_low",
        "scale": scale,
        "number_position": Gf.Vec3d(0.0, 9.6, 9.6), # ., Z, . ???
        "frame_position": Gf.Vec3d(0.0, 5.0, 1.0)
    }

    # 896
    houses_numbers_data["896"] = {
        "number_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/duplicated_01/duplicated_01",
        "frame_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/Kasa_Residential_01_low1/Kasa_Residential_01_low1",
        "scale": scale,
        "number_position": Gf.Vec3d(0.0, 5.0, 5.0),
        "frame_position": Gf.Vec3d(0.0, 5.0, 5.0)
    }

    # 895
    houses_numbers_data["895"] = {
        "number_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/duplicated_04/duplicated_04",
        "frame_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/Kasa_Residential_01_low2/Kasa_Residential_01_low2",
        "scale": scale,
        # "number_position": Gf.Vec3d(0.0, 5.0, 12.0),
        "number_position": Gf.Vec3d(0.0, 4.6, 9.6),
        "frame_position": Gf.Vec3d(0.0, 0.0, 1.0)
    }

    # 894
    houses_numbers_data["894"] = {
        "number_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/extracted_15/extracted_15",
        "frame_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/Kasa_Residential_01_low3/Kasa_Residential_01_low3",
        "scale": scale,
        "number_position": Gf.Vec3d(0.0, 0.0, 0.0),
        "frame_position": Gf.Vec3d(0.0, 0.0, 0.0)
    }

    # 893
    houses_numbers_data["893"] = {
        "number_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/extracted_16/extracted_16",
        "frame_prim_path": "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst/kasa_residential_house1_final/Kasa_Residential_low/Kasa_Residential_01_low4/Kasa_Residential_01_low4",
        "scale": scale,
        "number_position": Gf.Vec3d(0.0, 0.0, 0.0),
        "frame_position": Gf.Vec3d(0.0, 0.0, 0.0)
    }

    stage = get_context().get_stage()

    # Make the parent instance non-instanceable
    instance_path = "/World/layout/main/props/kasa_residential_house1_stand_in_40/kasa_residential_house1_stand_in_inst"

    instance_prim = stage.GetPrimAtPath(instance_path)

    if instance_prim.IsInstance():
        instance_prim.SetInstanceable(False)
        print(f"Made {instance_path} non-instanceable")
    else:
        print(f"Already non-instanceable")

    for house_number, house_number_config in houses_numbers_data.items():
        print(f"HOUSE NUMBER #{house_number}")
        number_prim = stage.GetPrimAtPath(house_number_config["number_prim_path"])
        frame_prim = stage.GetPrimAtPath(house_number_config["frame_prim_path"])
        new_scale = house_number_config["scale"]
        new_number_position = house_number_config["number_position"]
        new_frame_position = house_number_config["frame_position"]

        for prim in [number_prim, frame_prim]:
            print(f"PRIM {prim.GetName()}")
            if  prim.IsValid():
                # SCALE AND TRANSLATE
                xformable = UsdGeom.Xformable(prim)

                # Check for existing scale and translate
                scale_ops = [op for op in xformable.GetOrderedXformOps()
                             if op.GetOpType() == UsdGeom.XformOp.TypeScale]

                translate_ops = [op for op in xformable.GetOrderedXformOps()
                                 if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]

                # Scale
                xformable.AddScaleOp().Set(new_scale)
                print(f"Added scale x{scale[1]}")

                # Translate
                if prim == number_prim:
                    new_position = new_number_position
                else:
                    new_position = new_frame_position

                if translate_ops:
                    translate_ops[0].Set(new_position)
                    print(f"    Updated position to {new_position}")
                else:
                    xformable.AddTranslateOp().Set(new_position)
                    print(f"    Added position {new_position}")

def make_prim_noninstancable(prim_path: str):
    """Make the parent instance non-instanceable"""

    stage = get_context().get_stage()

    instance_prim = stage.GetPrimAtPath(prim_path)

    if instance_prim.IsInstance():
        instance_prim.SetInstanceable(False)
        print(f"Made {prim_path} non-instanceable")
    else:
        print(f"Already non-instanceable")

def replace_prim_with_obj():
    home_dir = os.path.expanduser("~")
    obj_path = os.path.join(home_dir, "Downloads", "Number_3.obj")
    # obj_path = os.path.join(home_dir, "Downloads", "Number_2020", "rp_posed_00178_29.usdz")

    number_prim_path = "/World/layout/main/props/kasa_residential_house2_stand_in_04/kasa_residential_house2_stand_in_inst/kasa_residential_house2_final/group1/duplicated_03/duplicated_03"

    make_prim_noninstancable("/World/layout/main/props/kasa_residential_house2_stand_in_04/kasa_residential_house2_stand_in_inst")

    stage = get_context().get_stage()

    # Get the existing prim
    existing_prim = stage.GetPrimAtPath(number_prim_path)

    if not existing_prim.IsValid():
        print(f"Prim not found: {number_prim_path}")
        return False

    # Delete the existing prim
    stage.RemovePrim(number_prim_path)

    # Create new Xform prim at the same path
    new_prim = UsdGeom.Xform.Define(stage, number_prim_path)

    # Add reference to the new USD file
    new_prim.GetPrim().GetReferences().AddReference(obj_path)

    # Apply transform
    xformable = UsdGeom.Xformable(new_prim)
    scale_ops = [op for op in xformable.GetOrderedXformOps()
                 if op.GetOpType() == UsdGeom.XformOp.TypeScale]

    translate_ops = [op for op in xformable.GetOrderedXformOps()
                     if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]

    rotate_ops = [op for op in xformable.GetOrderedXformOps()
                     if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]

    position = Gf.Vec3d(0.0, 0.0, 0.0)
    rotation = Gf.Vec3d(0.0, -90.0, 0.0)
    quat = Rotation.from_euler("XYZ", [rotation[0], rotation[2], rotation[2]], degrees=True).as_quat()
    rotation_quat = Gf.Quatd(quat[3], quat[0], quat[1], quat[2])
    scale = Gf.Vec3d(150.0, 150.0, 150.0)

    # Scale
    if scale_ops:
        scale_ops[0].Set(scale)
        print(f"    Updated scale to {scale}")
    else:
        xformable.AddScaleOp().Set(scale)
        print(f"    Added scale {scale}")

    # Rotate
    if rotate_ops:
        rotate_ops[0].Set(rotation)
        print(f"    Updated rotation to {rotation}")
    # else:
    #     xformable.AddOrientOp().Set(quat)
    #     print(f"    Added rotation {rotation}")

    # Translate
    if translate_ops:
        translate_ops[0].Set(position)
        print(f"    Updated position to {position}")
    else:
        xformable.AddTranslateOp().Set(position)
        print(f"    Added position {position}")

    print(f" + Replaced {number_prim_path} with {obj_path}")

    return True

# def replace_prim_references(prim_path, new_usd_path):
#     """
#     Replace the references of an existing prim with a new USD file.
#     This keeps the prim but changes what it references.
#     """
#     stage = get_context().get_stage()
#     prim = stage.GetPrimAtPath(prim_path)
#
#     if not prim.IsValid():
#         print(f"Prim not found: {prim_path}")
#         return False
#
#     # Clear all existing references
#     references = prim.GetReferences()
#     references.ClearReferences()
#
#     # Add new reference
#     references.AddReference(new_usd_path)
#
#     print(f"✓ Replaced references for {prim_path} with {new_usd_path}")
#     return True

def add_test_objects():
    """Add various physics objects around the drone for LiDAR testing"""
    print("Adding test objects for LiDAR detection...")

    # Create different materials for variety
    # metal_material = PhysicsMaterial(
    #     prim_path="/World/Materials/MetalMaterial",
    #     dynamic_friction=0.5,
    #     static_friction=0.6,
    #     restitution=0.3
    # )
    #
    # wood_material = PhysicsMaterial(
    #     prim_path="/World/Materials/WoodMaterial",
    #     dynamic_friction=0.7,
    #     static_friction=0.8,
    #     restitution=0.1
    # )

    # Fixed obstacles
    # Wall in front of the drone
    front_wall = FixedCuboid(
        prim_path="/World/FrontWall",
        name="front_wall",
        position=np.array([3.0, 0.0, 1.0]),  # 3m in front
        scale=np.array([0.2, 3.0, 2.0]),     # Thin wall, 3m wide, 2m tall
        color=np.array([0.8, 0.2, 0.2])      # Red color
    )

    # Wall behind the drone
    back_wall = FixedCuboid(
        prim_path="/World/BackWall",
        name="back_wall",
        position=np.array([-3.0, 0.0, 1.0]), # 3m behind
        scale=np.array([0.2, 3.0, 2.0]),     # Thin wall
        color=np.array([0.2, 0.8, 0.2])      # Green color
    )

    # Side barriers
    left_barrier = FixedCuboid(
        prim_path="/World/LeftBarrier",
        name="left_barrier",
        position=np.array([0.0, 4.0, 0.5]),  # 4m to the left
        scale=np.array([2.0, 0.2, 1.0]),     # Long barrier
        color=np.array([0.2, 0.2, 0.8])      # Blue color
    )

    right_barrier = FixedCuboid(
        prim_path="/World/RightBarrier",
        name="right_barrier",
        position=np.array([0.0, -4.0, 0.5]), # 4m to the right
        scale=np.array([2.0, 0.2, 1.0]),     # Long barrier
        color=np.array([0.8, 0.8, 0.2])      # Yellow color
    )

    # Dynamic objects (can be moved by physics)
    # Boxes at various distances
    box1 = DynamicCuboid(
        prim_path="/World/Box1",
        name="dynamic_box1",
        position=np.array([2.0, 1.5, 0.25]), # Front-left
        scale=np.array([0.5, 0.5, 0.5]),
        color=np.array([0.9, 0.5, 0.1]),     # Orange
        mass=1.0
    )

    box2 = DynamicCuboid(
        prim_path="/World/Box2",
        name="dynamic_box2",
        position=np.array([1.5, -2.0, 0.3]), # Front-right
        scale=np.array([0.6, 0.4, 0.6]),
        color=np.array([0.5, 0.1, 0.9]),     # Purple
        mass=1.5
    )

    box3 = DynamicCuboid(
        prim_path="/World/Box3",
        name="dynamic_box3",
        position=np.array([-1.5, 1.0, 0.4]), # Back-left
        scale=np.array([0.4, 0.7, 0.8]),
        color=np.array([0.1, 0.9, 0.5]),     # Light green
        mass=0.8
    )

    # Spheres for variety
    sphere1 = DynamicSphere(
        prim_path="/World/Sphere1",
        name="dynamic_sphere1",
        position=np.array([2.5, -1.0, 0.5]), # Front-right
        radius=0.3,
        color=np.array([0.9, 0.1, 0.1]),     # Bright red
        mass=0.5
    )

    sphere2 = DynamicSphere(
        prim_path="/World/Sphere2",
        name="dynamic_sphere2",
        position=np.array([-2.0, -1.5, 0.4]), # Back-right
        radius=0.25,
        color=np.array([0.1, 0.1, 0.9]),      # Bright blue
        mass=0.3
    )

    # Tall objects to test vertical FOV
    tall_pillar1 = FixedCuboid(
        prim_path="/World/TallPillar1",
        name="tall_pillar1",
        position=np.array([1.0, 2.5, 1.5]),  # Front-left, elevated
        scale=np.array([0.3, 0.3, 3.0]),     # Thin and tall
        color=np.array([0.6, 0.3, 0.6])      # Purple-gray
    )

    tall_pillar2 = FixedCuboid(
        prim_path="/World/TallPillar2",
        name="tall_pillar2",
        position=np.array([-0.5, -2.8, 1.2]), # Back-right, elevated
        scale=np.array([0.4, 0.4, 2.4]),      # Thick and tall
        color=np.array([0.3, 0.6, 0.3])       # Dark green
    )

    # Low objects to test minimum detection range
    low_box1 = FixedCuboid(
        prim_path="/World/LowBox1",
        name="low_box1",
        position=np.array([0.8, 0.8, 0.1]),  # Close to drone
        scale=np.array([0.3, 0.3, 0.2]),     # Low and small
        color=np.array([0.8, 0.8, 0.8])      # Gray
    )

    low_box2 = FixedCuboid(
        prim_path="/World/LowBox2",
        name="low_box2",
        position=np.array([-0.9, -0.7, 0.15]), # Close behind-right
        scale=np.array([0.4, 0.2, 0.3]),       # Low and rectangular
        color=np.array([0.4, 0.4, 0.4])        # Dark gray
    )

    print("Test objects added successfully!")

def add_objects_for_OA_test_mission1():
    wall_width = 4
    corridor_width = 7

    left_wall = FixedCuboid(
        prim_path="/World/LeftWall",
        name="left_wall",
        position=np.array([20, -corridor_width/2, 0.0]),
        scale=np.array([0.5, wall_width, 45]),
        color=np.array([0.0, 1.0, 0.0]),     # Green
    )

    right_wall = FixedCuboid(
        prim_path="/World/RightWall",
        name="right_wall",
        position=np.array([20, wall_width+corridor_width/2, 0.0]),
        scale=np.array([0.5, wall_width, 45]),
        color=np.array([1.0, 0.0, 1.0]),     # Purple
    )

def add_objects_for_OA_test_mission2():
    num_obstacles = 4
    width = 1
    margin = 3
    distance_x = 15
    even = True

    for i in range(num_obstacles):
        even = not even
        x = distance_x + i*2*margin+i*width/2
        y = even*(margin + width)

        color = [i*0.12, 1.0 - i*0.12, 0]

        FixedCuboid(
            prim_path=f"/World/Obstacle_{i}",
            name=f"obstacle_{i}",
            position=np.array([x, y, 0]),
            scale=np.array([width, width, 45]),
            color=np.array(color)
        )

def add_objects_for_OA_test_mission3():
    num_obstacles = 4
    width = 1
    margin = 3

    x = 15
    y = -(num_obstacles*width + (num_obstacles-1)*margin)

    for i in range(num_obstacles):
        x = x + margin + width/2
        for j in range(num_obstacles):
            y = y + margin + width/2
            color = [0.0, 0.7, 0.0] if (x+y)%2 else [0.1, 0.3, 0]

            FixedCuboid(
                prim_path=f"/World/Obstacle_{i}",
                name=f"obstacle_{i}",
                position=np.array([x, y, 0]),
                scale=np.array([width, width, 45]),
                color=np.array(color)
            )

def add_one_wall():
    front_wall = FixedCuboid(
        prim_path="/World/FrontObstacle",
        name="obstacle",
        position=np.array([40, 0.0, 0.0]),
        scale=np.array([0.5, 10.0, 45]),
        color=np.array([0.9, 0.1, 0.1]),     # Bright red
    )

def add_four_walls():
    add_one_wall()

    back_wall = FixedCuboid(
        prim_path="/World/BackObstacle",
        name="obstacle",
        position=np.array([-15, 0.0, 0.0]),
        scale=np.array([0.5, 10, 45]),
        color=np.array([0.1, 0.1, 0.9]),      # Bright blue
    )

    right_wall = FixedCuboid(
        prim_path="/World/RightObstacle",
        name="obstacle",
        position=np.array([0.0, 15.0, 0.0]),
        scale=np.array([10.0, 0.5, 45]),
        color=np.array([0.6, 0.3, 0.6])      # Purple-gray
    )

    left_wall = FixedCuboid(
        prim_path="/World/LeftObstacle",
        name="obstacle",
        position=np.array([0.0, -15.0, 0.0]),
        scale=np.array([10.0, 0.5, 45]),
        color=np.array([0.3, 0.6, 0.3]) # Dark green
    )
