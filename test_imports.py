#!/usr/bin/env python3
"""Test imports for Hunyuan3D-Part"""
import sys
import os

# Add paths properly
current_dir = os.path.dirname(os.path.abspath(__file__))
p3sam_dir = os.path.join(current_dir, "Hunyuan3D-Part", "P3-SAM")
xpart_dir = os.path.join(current_dir, "Hunyuan3D-Part", "XPart")

print(f"Current dir: {current_dir}")
print(f"P3-SAM dir: {p3sam_dir}")
print(f"P3-SAM dir exists: {os.path.exists(p3sam_dir)}")

# Test P3-SAM model import
try:
    sys.path.insert(0, p3sam_dir)
    from model import build_P3SAM, load_state_dict
    print("✓ P3-SAM model import successful")
except Exception as e:
    print(f"✗ P3-SAM model import failed: {e}")

# Test P3-SAM demo import
try:
    sys.path.insert(0, os.path.join(p3sam_dir, "demo"))
    from auto_mask import P3SAM, mesh_sam
    print("✓ P3-SAM demo import successful")
except Exception as e:
    print(f"✗ P3-SAM demo import failed: {e}")

# Test X-Part utils import
try:
    sys.path.insert(0, os.path.join(xpart_dir, "partgen"))
    from utils.misc import get_config_from_file
    print("✓ X-Part utils import successful")
except Exception as e:
    print(f"✗ X-Part utils import failed: {e}")

# Test X-Part pipeline import using dynamic function
try:
    from hy3dpart_nodes import _import_partformer_pipeline
    pipeline_class = _import_partformer_pipeline()
    if pipeline_class is not None:
        print("✓ X-Part pipeline import successful")
    else:
        print("✗ X-Part pipeline import failed: dynamic import returned None")
except Exception as e:
    print(f"✗ X-Part pipeline import failed: {e}")