from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, DynamicSphere
from omni.isaac.core.materials import PhysicsMaterial
from pxr import Usd, UsdGeom, Sdf, UsdLux
from omni.usd import get_context
import omni.isaac.core.utils.stage as stage_utils
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import os

# Isaac Sim RTX LiDAR imports
from pxr import Gf, Vt, Usd, UsdGeom, UsdPhysics, Sdf
# Near the top of add_objects.py, with other imports
from omni.isaac.core.utils.xforms import reset_xform_ops

# Add this function to the end of add_objects.py
# Add this function to the end of add_objects.py
from pxr import Gf, UsdGeom, UsdPhysics


def force_enable_collisions(stage):
    print("--- ENABLING COLLISIONS ---")

    # STEP 1: Un-instance everything
    # We collect paths first to avoid iterator invalidation when the stage changes
    print("1. Unpacking Instances (Parents)...")

    # We loop until no instanceable objects remain (handles nested instances)
    while True:
        instances_found = []
        for prim in stage.Traverse():
            if prim.IsInstanceable():
                instances_found.append(prim.GetPath())

        if not instances_found:
            break # Done!

        print(f"   Found {len(instances_found)} instances. Unpacking...")
        for path in instances_found:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                prim.SetInstanceable(False)

    # STEP 2: Force Physics on the exposed Meshes
    print("2. Applying Colliders to exposed meshes. It make take a while...")
    count = 0
    for prim in stage.Traverse():
        # We look for meshes that are NOT just leaves/grass
        if prim.IsA(UsdGeom.Mesh):

            # Force visibility (fix 'render' purpose)
            imageable = UsdGeom.Imageable(prim)
            if imageable.GetPurposeAttr().Get() != UsdGeom.Tokens.default_:
                imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)

            # Apply Collision
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)

            # Apply Accurate Mesh Collision (No approximation)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_api.CreateApproximationAttr().Set("none")
                count += 1

    print(f"SUCCESS: Enabled physics on {count} wall/structure meshes.")

    # STEP 3: Bake Physics
    print("3. Warming up physics engine...")
    for _ in range(60):
        self.world.step(render=False)
