import gradio as gr
import os
import sys
import argparse
import numpy as np
import trimesh
from pathlib import Path
import torch
import pytorch_lightning as pl
import spaces
import gc

# Import the VRAM-efficient model manager
from model_manager import model_manager

output_path = 'P3-SAM/results/gradio'
os.makedirs(output_path, exist_ok=True)

def is_supported_3d_file(filename):
    # 获取文件扩展名（小写），并去除开头的点
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.glb', '.ply', '.obj']

@spaces.GPU
def segment(mesh_path, postprocess=True, postprocess_threshold=0.95, seed=42):
    if mesh_path is None:
        gr.Warning("No Input Mesh")
        return None, None
    if not is_supported_3d_file(mesh_path):
        gr.Warning("Only support glb ply obj.")
        return None, None
    
    # Get P3-SAM model from model manager (loads on-demand)
    automask = model_manager.get_p3sam_model()
    
    # Set the target dtype for the model functions
    target_dtype = getattr(automask, '_target_dtype', torch.float32)
    if hasattr(automask, 'model'):
        setattr(automask.model, '_target_dtype', target_dtype)
    if hasattr(automask, 'model_parallel'):
        setattr(automask.model_parallel, '_target_dtype', target_dtype)
    
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    aabb, face_ids, mesh = automask.predict_aabb(mesh, seed=seed, is_parallel=False, post_process=postprocess, threshold=postprocess_threshold)
    
    # Free VRAM after segmentation
    model_manager.unload_all_models()
    
    color_map = {}
    unique_ids = np.unique(face_ids)
    for i in unique_ids:
        if i == -1:
            continue
        part_color = np.random.rand(3) * 255
        color_map[i] = part_color
    face_colors = []
    for i in face_ids:
        if i == -1:
            face_colors.append([0, 0, 0])
        else:
            face_colors.append(color_map[i])
    face_colors = np.array(face_colors).astype(np.uint8)
    mesh_save = mesh.copy()
    mesh_save.visual.face_colors = face_colors

    file_path = os.path.join(output_path, 'segment_mesh.glb')
    mesh_save.export(file_path)
    face_id_save_path = os.path.join(output_path, 'face_id.npy')
    np.save(face_id_save_path, face_ids)
    gr_state = [(aabb, mesh_path)]
    return file_path, face_id_save_path, gr_state

@spaces.GPU(duration=150)
def generate(mesh_path, seed=42, gr_state=None):
    if mesh_path is None:
        gr.Warning("No Input Mesh")
        if gr_state is not None:
            gr_state[0] = (None, None)
        return None, None, None
    if gr_state is None or gr_state[0][0] is None or mesh_path != gr_state[0][1]:
        gr.Warning("Please segment the mesh first")
        return None, None, None
    
    aabb = gr_state[0][0]
    
    # Get PartFormer pipeline from model manager (loads on-demand)
    pipeline = model_manager.get_partformer_pipeline()
    
    # Ensure deterministic behavior per request
    try:
        pl.seed_everything(int(seed), workers=True)
    except Exception:
        pl.seed_everything(2026, workers=True)
    
    additional_params = {"output_type": "trimesh"}
    obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = pipeline(
        mesh_path=mesh_path,
        aabb=aabb,
        octree_resolution=512,
        **additional_params,
    )
    
    # Free VRAM after generation
    model_manager.unload_all_models()
    
    # Export all results to temporary files for Gradio Model3D
    obj_path = os.path.join(output_path, 'obj_mesh.glb')
    out_bbox_path = os.path.join(output_path, 'out_bbox.glb')
    explode_path = os.path.join(output_path, 'explode.glb')
    obj_mesh.export(obj_path)
    out_bbox.export(out_bbox_path)
    explode_object.export(explode_path)
    return obj_path, out_bbox_path, explode_path

with gr.Blocks() as demo:
    gr.Markdown(
'''
# ☯️ Hunyuan3D Part:P3-SAM&XPart
This demo allows you to generate parts given a 3D model using Hunyuan3D-Part.
First segment the 3D model using P3-SAM and then generate parts using XPart.
Please upload glb ply or obj 3D model files.
Our examples are at the bottoms.
'''
    )
    with gr.Row():
        with gr.Column():
            # P3-SAM
            gr.Markdown(
'''
## P3-SAM: Native 3D Part Segmentation

[Paper](https://arxiv.org/abs/2509.06784) | [Project Page](https://murcherful.github.io/P3-SAM/) | [Code](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/P3-SAM/) | [Model](https://huggingface.co/tencent/Hunyuan3D-Part)

This is a demo of P3-SAM, a native 3D part segmentation method that can segment a mesh into different parts.
Input a mesh and push the "Segment" button to get the segmentation results.
'''
            )
            p3sam_button = gr.Button("Segment")
            p3sam_input = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Input Mesh")
            p3sam_output = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Segmentation Result")
            p3sam_face_id_output = gr.File(label='Face ID')
            p3sam_postprocess = gr.Checkbox(value=True, label="Post-processing")
            p3sam_postprocess_threshold = gr.Number(value=0.95, label="Post-processing Threshold")
            p3sam_seed = gr.Number(value=42, label="Random Seed")
            gr.Markdown(
'''
P3-SAM will clean your mesh. To get face-aligned labels, you can download the "Segmentation Result" and "Face ID".
You can also use the "Connectivity" and "Post-processing" options to control the behavior of the algorithm.
The "Post-processing" will merge the small parts according to the threshold. The smaller the threshold, the more parts will be merged.
'''
            )
            image_dump = gr.Image(label="Ref Image", visible=False)
        with gr.Column():
            # XPart
            gr.Markdown(
'''
## XPart: High-fidelity and Structure-coherent Shapede Composition  

[Paper](https://arxiv.org/abs/2509.08643) | [Project Page](https://yanxinhao.github.io/Projects/X-Part/) | [Code](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/XPart/) | [Model](https://huggingface.co/tencent/Hunyuan3D-Part)

This is a demo of the lite version of XPart, a high-fidelity and structure-coherent shape-decomposition method that can generate parts from a 3D model.
Input a mesh, segment it using P3-SAM on the left, and push the "Generate" button to get the generated parts.
'''         )
            xpart_button = gr.Button("Generate")
            xpart_output = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Generated Parts")
            xpart_output_bbox = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Gnerated Parts with BBox")
            xpart_output_exploded = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Exploded Object")
            xpart_seed = gr.Number(value=42, label="Random Seed")
        gr_state = gr.State(value=[(None, None)])
    with gr.Row():
        gr.Examples(examples=[
            ['P3-SAM/demo/assets/Female_Warrior.png'    , 'P3-SAM/demo/assets/Female_Warrior.glb'    ],
            ['P3-SAM/demo/assets/Suspended_Island.png'  , 'P3-SAM/demo/assets/Suspended_Island.glb'  ],
            ['P3-SAM/demo/assets/Beetle_Car.png'        , 'P3-SAM/demo/assets/Beetle_Car.glb'        ],
            ['XPart/data/Koi_Fish.png'                  , 'XPart/data/Koi_Fish.glb'                  ],
            ['XPart/data/Motorcycle.png'                , 'XPart/data/Motorcycle.glb'                ],
            ['XPart/data/Gundam.png'                    , 'XPart/data/Gundam.glb'                    ],
            ['XPart/data/Computer_Desk.png'             , 'XPart/data/Computer_Desk.glb'             ],
            ['XPart/data/Coffee_Machine.png'            , 'XPart/data/Coffee_Machine.glb'            ],
            ],
            inputs = [image_dump, p3sam_input],
        )

    p3sam_button.click(segment, inputs=[p3sam_input, p3sam_postprocess, p3sam_postprocess_threshold, p3sam_seed], outputs=[p3sam_output, p3sam_face_id_output, gr_state])
    xpart_button.click(generate, inputs=[p3sam_input, xpart_seed, gr_state], outputs=[xpart_output, xpart_output_bbox, xpart_output_exploded])


if __name__ == '__main__':
    try:
        demo.launch()
    finally:
        # Ensure all models are unloaded when the app exits
        model_manager.unload_all_models()
        print("App shutdown: All models unloaded")
