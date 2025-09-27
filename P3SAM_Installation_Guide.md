# P3-SAM Installation Troubleshooting Guide

## Issue: "P3-SAM functions not available" in ComfyUI

If you encounter this error in ComfyUI, follow these steps:

### 1. Check Dependencies

Run this in your ComfyUI environment:
```bash
# Navigate to ComfyUI directory
cd E:\Comfy-UI

# Activate ComfyUI environment
venv\Scripts\activate

# Check required packages
pip list | findstr "torch numpy trimesh spconv pytorch-lightning diffusers omegaconf"
```

### 2. Install Missing Dependencies

Install all required packages:
```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# AI/ML dependencies  
pip install pytorch-lightning diffusers transformers

# 3D processing
pip install trimesh scikit-learn fpsample numba

# Configuration and utilities
pip install omegaconf addict

# Sparse convolution (choose version matching your CUDA)
pip install spconv-cu118  # For CUDA 11.8
# OR
pip install spconv-cu121  # For CUDA 12.1
# OR  
pip install spconv-cu126  # For CUDA 12.6
```

### 3. Install Sonata (Required for P3-SAM)

```bash
# Clone and install Sonata
cd custom_nodes/ComfyUI-Hy3D-Part/Hunyuan3D-Part
git clone https://github.com/facebookresearch/sonata.git
cd sonata
pip install -e .
```

### 4. Verify Installation

Test the installation:
```bash
cd custom_nodes/ComfyUI-Hy3D-Part
python -c "
import sys
sys.path.insert(0, 'Hunyuan3D-Part/P3-SAM')
from model import build_P3SAM, load_state_dict
print('✓ P3-SAM model import successful')

sys.path.insert(0, 'Hunyuan3D-Part/P3-SAM/demo')  
from auto_mask import P3SAM, mesh_sam
print('✓ P3-SAM demo import successful')
"
```

### 5. Common Issues and Solutions

#### Issue: "No module named 'spconv'"
**Solution:** Install the correct spconv version for your CUDA:
```bash
# Check CUDA version
nvidia-smi

# Install matching spconv
pip install spconv-cu118  # Replace with your CUDA version
```

#### Issue: "No module named 'sonata'"
**Solution:** Install Sonata properly:
```bash
cd Hunyuan3D-Part/sonata
pip install -e .
```

#### Issue: "No module named 'utils.misc'"
**Solution:** This is handled automatically by the enhanced import system. If it persists:
```bash
# Ensure XPart path is correct
cd Hunyuan3D-Part/XPart/partgen
python -c "from utils.misc import get_config_from_file; print('✓ XPart utils work')"
```

### 6. Restart ComfyUI

After installing dependencies:
1. **Stop ComfyUI completely**
2. **Restart ComfyUI**
3. **Test the Hunyuan3D nodes**

### 7. Enhanced Error Handling

The integration now includes:
- ✅ **Runtime import fallback** - Attempts to load P3-SAM at node execution time
- ✅ **Minimal implementation** - Provides basic functionality even if full import fails
- ✅ **Clear error messages** - Tells you exactly what's missing
- ✅ **Graceful degradation** - Node doesn't crash ComfyUI startup

### 8. Verification Commands

Run these to verify everything works:

```bash
# Test P3-SAM integration
python -c "
from hy3dpart_nodes import Hy3DPartSegmentation
node = Hy3DPartSegmentation()
print('✓ P3-SAM node created successfully')
"

# Test full integration
python test_integration.py
```

### 9. If Problems Persist

1. **Check ComfyUI logs** for specific error messages
2. **Ensure GPU memory** is sufficient (8GB+ recommended)
3. **Verify CUDA compatibility** between PyTorch and spconv
4. **Try minimal mode** - the integration provides fallbacks for basic functionality

### 10. Contact Information

If you continue to have issues:
- Check the [Hunyuan3D-Part GitHub Issues](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/issues)
- Verify your environment matches the requirements
- Consider using the minimal implementation for basic testing

The integration is designed to be robust and provide helpful error messages to guide you through any remaining issues.