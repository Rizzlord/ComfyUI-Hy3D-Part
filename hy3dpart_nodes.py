import os
import sys
import torch
import numpy as np
import trimesh
import pytorch_lightning as pl
import folder_paths
from pathlib import Path

# --- Path Setup ---
# This block ensures that modules from the 'Hunyuan3D-Part' subdirectory can be imported.
current_node_dir = os.path.dirname(os.path.realpath(__file__))
hunyuan_dir = os.path.join(current_node_dir, "Hunyuan3D-Part")

# Add the subdirectory and its nested modules to the Python path
sys.path.insert(0, hunyuan_dir)
sys.path.insert(0, os.path.join(hunyuan_dir, 'P3-SAM'))
sys.path.insert(0, os.path.join(hunyuan_dir, 'XPart'))
sys.path.insert(0, os.path.join(hunyuan_dir, 'XPart', 'partgen'))


# --- Model Path Patching ---
base_weights_dir = os.path.abspath(os.path.join(current_node_dir, 'weights'))
p3sam_model_path = os.path.join(base_weights_dir, 'p3sam', 'p3sam.safetensors')
partformer_model_path = os.path.join(base_weights_dir, 'Hunyuan3D-Part')

try:
    import P3_SAM.model as p3sam_model_module
    original_load_state_dict = p3sam_model_module.load_state_dict

    def patched_load_state_dict(self, ckpt_path=None, state_dict=None, strict=True, assign=False, ignore_seg_mlp=False, ignore_seg_s2_mlp=False, ignore_iou_mlp=False):
        if ckpt_path is None and state_dict is None:
            print(f"Hy3D-Part Node: Redirecting P3-SAM model load to local path: {p3sam_model_path}")
            if not os.path.exists(p3sam_model_path):
                raise FileNotFoundError(f"P3-SAM model not found at {p3sam_model_path}. Please ensure it is placed correctly.")
            ckpt_path = p3sam_model_path
        return original_load_state_dict(self, ckpt_path, state_dict, strict, assign, ignore_seg_mlp, ignore_seg_s2_mlp, ignore_iou_mlp)

    p3sam_model_module.load_state_dict = patched_load_state_dict
    print("Hy3D-Part Node: P3-SAM model loading patched successfully.")
except Exception as e:
    print(f"Hy3D-Part Node: Failed to patch P3-SAM model loading: {e}")


try:
    from partgen.partformer_pipeline import PartFormerPipeline
    original_from_pretrained = PartFormerPipeline.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, model_path="tencent/Hunyuan3D-Part", dtype=torch.float32, device="cuda", **kwargs):
        if model_path == "tencent/Hunyuan3D-Part":
            print(f"Hy3D-Part Node: Redirecting PartFormer model load to local path: {partformer_model_path}")
            if not os.path.exists(partformer_model_path):
                 raise FileNotFoundError(f"PartFormer model directory not found at {partformer_model_path}. Please ensure the model subdirectories ('model', 'conditioner', etc.) are placed there.")
            model_path = partformer_model_path
        return original_from_pretrained(model_path=model_path, dtype=dtype, device=device, **kwargs)

    PartFormerPipeline.from_pretrained = patched_from_pretrained
    print("Hy3D-Part Node: PartFormer pipeline loading patched successfully.")
except Exception as e:
    print(f"Hy3D-Part Node: Failed to patch PartFormer pipeline loading: {e}")

from model_manager import model_manager

# --- ComfyUI Nodes ---

class LoadTrimeshFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "E:\\path\\to\\your\\model.glb"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "load_mesh"
    CATEGORY = "Hy3D-Part/IO"

    def load_mesh(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.glb', '.ply', '.obj']:
             raise ValueError("Unsupported file type. Please use .glb, .ply, or .obj")
        
        mesh = trimesh.load(file_path, force='mesh', process=False)
        print(f"Hy3D-Part Node: Loaded trimesh object from {file_path}")
        return (mesh,)


class SaveTrimeshFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "Hy3D"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_mesh"
    CATEGORY = "Hy3D-Part/IO"
    OUTPUT_NODE = True

    def save_mesh(self, mesh, filename_prefix):
        output_dir = os.path.join(folder_paths.get_output_directory(), "Hy3D-Part")
        os.makedirs(output_dir, exist_ok=True)
        
        full_prefix = folder_paths.get_timestamped_filename(filename_prefix)
        file_path = os.path.join(output_dir, f"{full_prefix}.glb")
        
        mesh.export(file_path)
        print(f"Hy3D-Part Node: Saved trimesh object to {file_path}")
        
        return (file_path,)


class P3_SAM_Segmenter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH", ),
                "postprocess": ("BOOLEAN", {"default": True}),
                "postprocess_threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "HY3D_AABB", "TRIMESH")
    RETURN_NAMES = ("segmented_mesh", "aabb_data", "processed_mesh")
    FUNCTION = "segment"
    CATEGORY = "Hy3D-Part"

    def segment(self, mesh, postprocess, postprocess_threshold, seed):
        print("Hy3D-Part Node: Starting P3-SAM segmentation.")
        
        automask = model_manager.get_p3sam_model()
        
        target_dtype = getattr(automask, '_target_dtype', torch.float32)
        if hasattr(automask, 'model'):
            setattr(automask.model, '_target_dtype', target_dtype)
        if hasattr(automask, 'model_parallel'):
            setattr(automask.model_parallel, '_target_dtype', target_dtype)
            
        aabb, face_ids, processed_mesh = automask.predict_aabb(mesh, seed=seed, is_parallel=False, post_process=postprocess, threshold=postprocess_threshold)
        
        model_manager.unload_all_models()
        print("Hy3D-Part Node: P3-SAM model unloaded from VRAM.")
        
        color_map = {}
        unique_ids = np.unique(face_ids)
        for i in unique_ids:
            if i == -1:
                continue
            part_color = np.random.rand(3) * 255
            color_map[i] = part_color
        
        face_colors = []
        for i in face_ids:
            face_colors.append(color_map.get(i, [0, 0, 0]))
        
        face_colors = np.array(face_colors).astype(np.uint8)
        segmented_mesh_vis = processed_mesh.copy()
        segmented_mesh_vis.visual.face_colors = face_colors

        print(f"Hy3D-Part Node: Segmentation complete.")
        
        return (segmented_mesh_vis, aabb, processed_mesh)


class XPart_Generator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "processed_mesh": ("TRIMESH", ),
                "aabb_data": ("HY3D_AABB", ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "octree_resolution": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64}),
            }
        }
    
    RETURN_TYPES = ("TRIMESH", "TRIMESH", "TRIMESH",)
    RETURN_NAMES = ("generated_parts", "bbox_view", "exploded_view",)
    FUNCTION = "generate"
    CATEGORY = "Hy3D-Part"

    def generate(self, processed_mesh, aabb_data, seed, octree_resolution):
        print("Hy3D-Part Node: Starting XPart generation.")

        pipeline = model_manager.get_partformer_pipeline()
        
        try:
            pl.seed_everything(int(seed), workers=True)
        except Exception:
            pl.seed_everything(2026, workers=True)
        
        additional_params = {"output_type": "trimesh"}
        obj_mesh, (out_bbox, _, explode_object) = pipeline(
            mesh=processed_mesh,
            aabb=aabb_data,
            octree_resolution=octree_resolution,
            **additional_params,
        )
        
        model_manager.unload_all_models()
        print("Hy3D-Part Node: PartFormer model unloaded from VRAM.")
        
        print(f"Hy3D-Part Node: Generation complete.")

        return (obj_mesh, out_bbox, explode_object)


# --- Node Mappings ---

NODE_CLASS_MAPPINGS = {
    "Hy3D_LoadTrimeshFile": LoadTrimeshFile,
    "Hy3D_SaveTrimeshFile": SaveTrimeshFile,
    "Hy3D_P3_SAM_Segmenter": P3_SAM_Segmenter,
    "Hy3D_XPart_Generator": XPart_Generator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3D_LoadTrimeshFile": "Load Trimesh File (Hy3D)",
    "Hy3D_SaveTrimeshFile": "Save Trimesh File (Hy3D)",
    "Hy3D_P3_SAM_Segmenter": "P3-SAM Segmenter (Hy3D)",
    "Hy3D_XPart_Generator": "XPart Generator (Hy3D)"
}