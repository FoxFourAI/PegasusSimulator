from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, DynamicSphere
from omni.isaac.core.materials import PhysicsMaterial
import numpy as np

def add_test_objects():
    """Add various physics objects around the drone for LiDAR testing"""
    print("Adding test objects for LiDAR detection...")

    # Create different materials for variety
    metal_material = PhysicsMaterial(
        prim_path="/World/Materials/MetalMaterial",
        dynamic_friction=0.5,
        static_friction=0.6,
        restitution=0.3
    )

    wood_material = PhysicsMaterial(
        prim_path="/World/Materials/WoodMaterial",
        dynamic_friction=0.7,
        static_friction=0.8,
        restitution=0.1
    )

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

    # Apply materials to some objects for realistic LiDAR reflectivity
    try:
        # Apply metal material to some objects
        metal_material.apply_to_prim(front_wall.prim)
        metal_material.apply_to_prim(sphere1.prim)

        # Apply wood material to others
        wood_material.apply_to_prim(box1.prim)
        wood_material.apply_to_prim(tall_pillar1.prim)
    except Exception as e:
        print(f"Warning: Could not apply materials to objects: {e}")

    print("Test objects added successfully!")
