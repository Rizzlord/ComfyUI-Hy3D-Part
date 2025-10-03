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
                "point_num": ("INT", {"default": 55000, "min": 1000, "max": 100000, "step": 1000}),
                "prompt_num": ("INT", {"default": 400, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "HY3D_AABB", "TRIMESH")
    RETURN_NAMES = ("segmented_mesh", "aabb_data", "processed_mesh")
    FUNCTION = "segment"
    CATEGORY = "Hy3D-Part"

    def segment(self, mesh, postprocess, postprocess_threshold, seed, point_num, prompt_num):
        print("Hy3D-Part Node: Starting P3-SAM segmentation.")
        
        automask = model_manager.get_p3sam_model()
        
        target_dtype = getattr(automask, '_target_dtype', torch.float32)
        if hasattr(automask, 'model'):
            setattr(automask.model, '_target_dtype', target_dtype)
        if hasattr(automask, 'model_parallel'):
            setattr(automask.model_parallel, '_target_dtype', target_dtype)
            
        aabb, face_ids, processed_mesh = automask.predict_aabb(
            mesh,
            seed=seed,
            is_parallel=False,
            post_process=postprocess,
            threshold=postprocess_threshold,
            point_num=point_num,
            prompt_num=prompt_num,
        )
        
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


class ColorSeperation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_mesh": ("TRIMESH",),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "ignore_black": ("BOOLEAN", {"default": False}),
                "min_faces": ("INT", {"default": 1, "min": 1, "max": 1000000, "step": 1}),
                "clamp_colors": ("BOOLEAN", {"default": True}),
                "clamp_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "separate_and_fill"
    CATEGORY = "Hy3D-Part"

    def separate_and_fill(self, segmented_mesh, fill_holes, ignore_black, min_faces, clamp_colors, clamp_threshold):
        if segmented_mesh is None:
            raise ValueError("No mesh provided for color separation.")

        mesh = segmented_mesh.copy()
        face_colors = self._extract_face_colors(mesh)
        if clamp_colors:
            face_colors = self._clamp_colors(face_colors, clamp_threshold)

        unique_colors, inverse = np.unique(face_colors, axis=0, return_inverse=True)
        parts = []
        parts_face_colors = []
        ignored_black_faces = []

        for color_idx, color in enumerate(unique_colors):
            face_indices = np.where(inverse == color_idx)[0]
            if face_indices.size == 0:
                continue

            if ignore_black and np.all(color[:3] == 0):
                ignored_black_faces.append(face_indices)
                continue

            small_part = face_indices.size < min_faces

            submesh = mesh.submesh([face_indices], append=True, repair=False)
            if isinstance(submesh, list):
                if not submesh:
                    continue
                submesh = submesh[0]

            submesh = submesh.copy()
            if submesh.faces.size == 0:
                continue

            rgba_color = self._ensure_rgba(color)

            if fill_holes and not small_part:
                try:
                    trimesh.repair.fill_holes(submesh)
                except Exception as exc:
                    print(f"Hy3D-Part Node: Hole filling failed for color {color_idx}: {exc}")
                else:
                    trimesh.repair.fix_normals(submesh, multibody=True)

            part_colors = np.tile(rgba_color, (submesh.faces.shape[0], 1))
            submesh.visual.face_colors = part_colors

            parts.append(submesh)
            parts_face_colors.append(part_colors)

        if ignored_black_faces:
            if len(ignored_black_faces) == 1:
                leftover_indices = ignored_black_faces[0]
            else:
                leftover_indices = np.concatenate(ignored_black_faces)
            leftover_indices = np.asarray(leftover_indices, dtype=np.int64)
            leftover_indices = np.unique(leftover_indices)
            leftover_submesh = mesh.submesh([leftover_indices], append=True, repair=False)
            if isinstance(leftover_submesh, list):
                leftover_submesh = leftover_submesh[0] if leftover_submesh else None
            if leftover_submesh is not None:
                leftover_submesh = leftover_submesh.copy()
                if leftover_submesh.faces.size:
                    if fill_holes:
                        try:
                            trimesh.repair.fill_holes(leftover_submesh)
                        except Exception as exc:
                            print(f"Hy3D-Part Node: Hole filling failed for leftover faces: {exc}")
                        else:
                            trimesh.repair.fix_normals(leftover_submesh, multibody=True)
                    leftover_colors = face_colors[leftover_indices].copy()
                    leftover_submesh.visual.face_colors = leftover_colors
                    parts.append(leftover_submesh)
                    parts_face_colors.append(leftover_colors)

        if not parts:
            raise ValueError("No color-separated parts were produced. Check segmentation colors and settings.")

        color_tiles = parts_face_colors
        recombined_mesh = trimesh.util.concatenate(parts)
        recombined_mesh.visual.face_colors = np.vstack(color_tiles)
        recombined_mesh.metadata = (recombined_mesh.metadata or {})
        recombined_mesh.metadata["color_parts"] = parts

        return (recombined_mesh,)

    @staticmethod
    def _clamp_colors(colors, threshold):
        threshold = float(np.clip(threshold, 0.0, 1.0))
        colors_uint8 = ColorSeperation._to_uint8(colors)
        clamped = colors_uint8.copy()
        rgb = clamped[:, :3].astype(np.float32) / 255.0
        rgb = (rgb >= threshold).astype(np.uint8) * 255
        clamped[:, :3] = rgb
        return clamped

    @staticmethod
    def _to_uint8(colors):
        arr = np.asarray(colors)
        if arr.dtype == np.uint8:
            return arr.astype(np.uint8, copy=False)
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = np.clip(np.round(arr * 255.0), 0, 255)
        else:
            arr = np.clip(np.round(arr), 0, 255)
        return arr.astype(np.uint8)

    @staticmethod
    def _ensure_rgba(color):
        color = np.asarray(color, dtype=np.uint8).reshape(-1)
        if color.shape[0] == 4:
            return color
        if color.shape[0] == 3:
            return np.append(color, 255).astype(np.uint8)
        raise ValueError("Unexpected color dimensionality encountered during separation.")

    @staticmethod
    def _extract_face_colors(mesh):
        visual = getattr(mesh, "visual", None)
        if visual is None:
            raise ValueError("Mesh does not contain visual color information.")

        face_colors = getattr(visual, "face_colors", None)
        if face_colors is not None and len(face_colors):
            return ColorSeperation._to_uint8(face_colors[:, :4])

        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is not None and len(vertex_colors):
            vertex_colors = ColorSeperation._to_uint8(vertex_colors[:, :4])
            face_indices = mesh.faces
            face_vertex_colors = vertex_colors[face_indices]
            return face_vertex_colors[:, 0, :]

        raise ValueError("Mesh does not provide face or vertex colors required for separation.")


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
    "Hy3D_ColorSeperation": ColorSeperation,
    "Hy3D_XPart_Generator": XPart_Generator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3D_LoadTrimeshFile": "Load Trimesh File (Hy3D)",
    "Hy3D_SaveTrimeshFile": "Save Trimesh File (Hy3D)",
    "Hy3D_P3_SAM_Segmenter": "P3-SAM Segmenter (Hy3D)",
    "Hy3D_ColorSeperation": "Color Separation (Hy3D)",
    "Hy3D_XPart_Generator": "XPart Generator (Hy3D)"
}