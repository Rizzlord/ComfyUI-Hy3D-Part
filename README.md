# ComfyUI-Hy3D-Part

ComfyUI custom nodes for Tencent's Hunyuan3D-Part pipeline, providing state-of-the-art 3D part segmentation and generation capabilities.

## Features

### P3-SAM: Native 3D Part Segmentation
- Automatic 3D mesh segmentation into semantic parts
- Point-promptable segmentation with IoU prediction
- Robust handling of complex 3D objects
- Integration with ComfyUI's TRIMESH type

### X-Part: High-fidelity Shape Decomposition
- Structure-coherent part generation
- Controllable generation with bounding box prompts
- High geometric fidelity output
- Exploded view generation for visualization

## Installation

1. **Clone this repository** into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-repo/ComfyUI-Hy3D-Part.git
```

2. **Clone the Hunyuan3D-Part repository** inside this directory:
```bash
cd ComfyUI-Hy3D-Part/
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git
```

3. **Install dependencies**:
```bash
pip install trimesh torch numpy fpsample
pip install viser gradio  # Optional for interactive demos
```

4. **Install Sonata dependencies** (required for both P3-SAM and X-Part):
Follow the installation instructions from [Sonata repository](https://github.com/facebookresearch/sonata)

5. **Install X-Part specific dependencies**:
```bash
cd Hunyuan3D-Part/XPart/
pip install -r requirements.txt
```

6. **Set up P3-SAM chamfer distance** (optional but recommended):
```bash
cd Hunyuan3D-Part/P3-SAM/utils/chamfer3D/
python setup.py install
```

## Nodes

### Hy3DPartSegmentation
**P3-SAM: Native 3D Part Segmentation**

Segments a 3D mesh into semantic parts using the P3-SAM model.

**Inputs:**
- `trimesh`: Input 3D mesh (TRIMESH type)
- `point_num`: Number of points to sample (default: 100000)
- `prompt_num`: Number of prompt points (default: 400)
- `prompt_bs`: Batch size for prompts (default: 32)
- `threshold`: Post-processing threshold (default: 0.95)
- `seed`: Random seed (default: 42)
- `post_process`: Enable post-processing (default: True)
- `clean_mesh_flag`: Clean input mesh (default: True)
- `model_path`: Optional custom model path

**Outputs:**
- `segmented_mesh`: Mesh with part segmentation
- `segmentation_info`: Information about segmentation results
- `part_bboxes`: List of part bounding boxes

### Hy3DPartGeneration
**X-Part: High-fidelity Structure-coherent Shape Decomposition**

Generates detailed part meshes from segmentation data.

**Inputs:**
- `trimesh`: Input 3D mesh (TRIMESH type)
- `num_inference_steps`: Diffusion sampling steps (default: 50)
- `guidance_scale`: Guidance scale for generation (default: -1.0)
- `octree_resolution`: Resolution for mesh extraction (default: 512)
- `seed`: Random seed (default: 42)
- `config_path`: Optional custom config path
- `part_bboxes`: Optional part bounding boxes from segmentation

**Outputs:**
- `generated_parts`: Generated part meshes (Scene)
- `bbox_visualization`: Visualization with bounding boxes
- `exploded_view`: Exploded view of parts
- `generation_info`: Information about generation results

### Hy3DPartPipeline
**Combined P3-SAM + X-Part Pipeline**

Performs both segmentation and part generation in one node.

**Inputs:** Combination of P3-SAM and X-Part parameters

**Outputs:**
- `segmented_mesh`: Segmented input mesh
- `generated_parts`: Generated part meshes
- `bbox_visualization`: Bounding box visualization
- `exploded_view`: Exploded view
- `pipeline_info`: Pipeline execution information
- `part_bboxes`: Part bounding box data

### Hy3DExportParts
**Export Individual Parts**

Export parts as separate GLB files.

**Inputs:**
- `trimesh_scene`: Scene with multiple parts
- `output_path`: Output directory path
- `filename_prefix`: Prefix for exported files

**Outputs:**
- `export_info`: Export status information
- `exported_files`: List of exported file paths

## Integration with Blender Tools

These nodes work seamlessly with existing Blender tool nodes:

### Example Workflow
1. **Load/Generate** initial mesh
2. **BlenderDecimate** → Reduce mesh complexity if needed
3. **Hy3DPartSegmentation** → Segment into parts
4. **Hy3DPartGeneration** → Generate detailed parts
5. **BlenderExportGLB** → Export final results

### Workflow Benefits
- **Preprocessing**: Use BlenderDecimate to optimize mesh complexity before segmentation
- **Post-processing**: Use BlenderExportGLB for clean final exports
- **Quality Control**: Apply mesh cleaning operations between steps

## Model Downloads

The nodes will automatically download required models from HuggingFace:

- **P3-SAM model**: `tencent/Hunyuan3D-Part` (p3sam.ckpt)
- **X-Part model**: `tencent/Hunyuan3D-Part` (xpart.pt)

Models are cached locally after first download.

## Examples

### Basic Part Segmentation
```
Input Mesh → Hy3DPartSegmentation → Segmented Mesh + Part Info
```

### Full Pipeline
```
Input Mesh → Hy3DPartPipeline → Generated Parts + Visualizations
```

### With Blender Integration
```
Input Mesh → BlenderDecimate → Hy3DPartPipeline → BlenderExportGLB
```

## Performance Notes

- **GPU Required**: CUDA-capable GPU recommended for both P3-SAM and X-Part
- **Memory Usage**: Large meshes may require 8GB+ VRAM
- **Processing Time**: 
  - Segmentation: 1-5 minutes depending on mesh complexity
  - Generation: 2-10 minutes depending on inference steps and resolution

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure Hunyuan3D-Part is cloned in the correct location
2. **CUDA errors**: Verify PyTorch CUDA installation
3. **Sonata errors**: Follow Sonata installation guide carefully
4. **Model download fails**: Check internet connection and HuggingFace access

### Debug Mode

Set environment variable for verbose output:
```bash
export HUNYUAN_DEBUG=1
```

## Citation

If you use this integration in your work, please cite the original papers:

**P3-SAM:**
```bibtex
@article{ma2025p3sam,
  title={P3-sam: Native 3d part segmentation},
  author={Ma, Changfeng and Li, Yang and Yan, Xinhao and Xu, Jiachen and Yang, Yunhan and Wang, Chunshi and Zhao, Zibo and Guo, Yanwen and Chen, Zhuo and Guo, Chunchao},
  journal={arXiv preprint arXiv:2509.06784},
  year={2025}
}
```

**X-Part:**
```bibtex
@article{yan2025xpart,
  title={X-Part: high fidelity and structure coherent shape decomposition},
  author={Yan, Xinhao and Xu, Jiachen and Li, Yang and Ma, Changfeng and Yang, Yunhan and Wang, Chunshi and Zhao, Zibo and Lai, Zeqiang and Zhao, Yunfei and Chen, Zhuo and others},
  journal={arXiv preprint arXiv:2509.08643},
  year={2025}
}
```

## License

This integration follows the licenses of the original Hunyuan3D-Part project. Please refer to the original repository for license details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.