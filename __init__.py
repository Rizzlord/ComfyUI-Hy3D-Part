"""
ComfyUI-Hy3D-Part: Hunyuan3D Part Generation and Segmentation for ComfyUI

This module integrates Tencent's Hunyuan3D-Part pipeline with ComfyUI, providing:
- P3-SAM: Native 3D part segmentation
- X-Part: High-fidelity and structure-coherent shape decomposition

Requirements:
- trimesh
- torch
- numpy
- fpsample
- gradio (optional)
- viser (optional)
"""

import os
import sys

# Add the Hunyuan3D-Part directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
hunyuan_dir = os.path.join(current_dir, "Hunyuan3D-Part")
p3sam_dir = os.path.join(hunyuan_dir, "P3-SAM")
xpart_dir = os.path.join(hunyuan_dir, "XPart")

if hunyuan_dir not in sys.path:
    sys.path.insert(0, hunyuan_dir)
if p3sam_dir not in sys.path:
    sys.path.insert(0, p3sam_dir)
if xpart_dir not in sys.path:
    sys.path.insert(0, xpart_dir)

try:
    from .hy3dpart_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    # Export for ComfyUI
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    
except ImportError as e:
    print(f"Warning: Failed to import Hunyuan3D-Part nodes: {e}")
    print("Please ensure all dependencies are installed:")
    print("- pip install trimesh torch numpy fpsample")
    print("- Install required packages from Hunyuan3D-Part/XPart/requirements.txt")
    
    # Provide empty mappings to prevent ComfyUI from failing
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Metadata
WEB_DIRECTORY = "./web"
__version__ = "1.0.0"