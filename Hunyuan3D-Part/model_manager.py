import torch
import gc
import threading
from typing import Optional, Dict, Any, Union
import sys
from pathlib import Path
import warnings
from enum import Enum
import json
import os

# Import the model classes
sys.path.append('P3-SAM')
from demo.auto_mask import AutoMask
sys.path.append('XPart')
from partgen.partformer_pipeline import PartFormerPipeline
from partgen.utils.misc import get_config_from_file


class PrecisionMode(Enum):
    """Enumeration for different precision modes"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"  # Experimental support


def load_precision_from_config() -> PrecisionMode:
    """Load precision mode from config file if it exists"""
    config_path = Path(__file__).parent / "precision_config.json"
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                precision_str = config.get('precision_mode', 'fp16')
                return PrecisionMode(precision_str.lower())
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Failed to load precision config: {e}. Using default FP16.")
    
    return PrecisionMode.FP16  # Default to FP16 for better VRAM efficiency


def save_precision_to_config(precision_mode: PrecisionMode):
    """Save precision mode to config file"""
    config_path = Path(__file__).parent / "precision_config.json"
    
    try:
        config = {'precision_mode': precision_mode.value}
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save precision config: {e}")


class VRAMEfficientModelManager:
    """
    A VRAM-efficient model manager that loads models on-demand and properly cleans them up.
    Only one model is kept in memory at a time to minimize VRAM usage.
    Supports multiple precision modes for further VRAM reduction.
    """
    
    def __init__(self, precision_mode: Union[str, PrecisionMode] = PrecisionMode.FP32):
        self._current_model_type = None
        self._p3sam_model = None
        self._partformer_pipeline = None
        self._lock = threading.Lock()
        
        # Set precision mode
        if isinstance(precision_mode, str):
            precision_mode = PrecisionMode(precision_mode.lower())
        self.precision_mode = precision_mode
        
        # Configure torch dtype based on precision mode
        self.torch_dtype = self._get_torch_dtype()
        
        print(f"Model Manager initialized with precision mode: {self.precision_mode.value}")
        print(f"Using torch dtype: {self.torch_dtype}")
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get the appropriate torch dtype based on precision mode"""
        dtype_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.FP8: torch.float16  # Always use FP16 for FP8 until stable support
        }
        
        selected_dtype = dtype_map.get(self.precision_mode, torch.float32)
        
        # Check hardware compatibility for BF16
        if self.precision_mode == PrecisionMode.BF16 and not torch.cuda.is_bf16_supported():
            warnings.warn("BFloat16 not supported on this GPU, falling back to FP16")
            selected_dtype = torch.float16
        
        # FP8 note
        if self.precision_mode == PrecisionMode.FP8:
            print("Note: FP8 mode uses FP16 internally for maximum compatibility")
            
        return selected_dtype
    
    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply quantization to the model based on precision mode"""
        if self.precision_mode == PrecisionMode.FP32:
            return model  # No quantization needed
        
        try:
            # For FP8, we currently fall back to FP16 due to limited PyTorch support
            # This ensures consistency across all model components
            target_dtype = self.torch_dtype
            if self.precision_mode == PrecisionMode.FP8:
                print("FP8 quantization requested - using FP16 for stability")
                target_dtype = torch.float16  # Force FP16 instead of FP8 for now
            
            # Recursively convert all parameters and buffers to target dtype
            def convert_recursive(module):
                for param in module.parameters():
                    param.data = param.data.to(dtype=target_dtype)
                for buffer in module.buffers():
                    if buffer.dtype.is_floating_point:
                        buffer.data = buffer.data.to(dtype=target_dtype)
                for child in module.children():
                    convert_recursive(child)
            
            convert_recursive(model)
            
            # Also use the simplified .to() method as a backup
            model = model.to(dtype=target_dtype)
            
            print(f"Model quantized to {target_dtype} (requested: {self.precision_mode.value})")
            return model
            
        except Exception as e:
            warnings.warn(f"Quantization to {self.precision_mode.value} failed: {e}. Using FP32.")
            return model.to(dtype=torch.float32)
        
    def _clear_vram(self):
        """Force clear VRAM and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def _unload_current_model(self):
        """Unload the currently loaded model and free VRAM"""
        if self._current_model_type == "p3sam" and self._p3sam_model is not None:
            # Move model to CPU and delete
            if hasattr(self._p3sam_model, 'model') and self._p3sam_model.model is not None:
                self._p3sam_model.model.cpu()
                del self._p3sam_model.model
            if hasattr(self._p3sam_model, 'model_parallel') and self._p3sam_model.model_parallel is not None:
                self._p3sam_model.model_parallel.cpu()
                del self._p3sam_model.model_parallel
            del self._p3sam_model
            self._p3sam_model = None
            
        elif self._current_model_type == "partformer" and self._partformer_pipeline is not None:
            # Move pipeline components to CPU and delete
            if hasattr(self._partformer_pipeline, 'vae') and self._partformer_pipeline.vae is not None:
                self._partformer_pipeline.vae.cpu()
                del self._partformer_pipeline.vae
            if hasattr(self._partformer_pipeline, 'model') and self._partformer_pipeline.model is not None:
                self._partformer_pipeline.model.cpu()
                del self._partformer_pipeline.model
            if hasattr(self._partformer_pipeline, 'conditioner') and self._partformer_pipeline.conditioner is not None:
                self._partformer_pipeline.conditioner.cpu()
                del self._partformer_pipeline.conditioner
            if hasattr(self._partformer_pipeline, 'scheduler'):
                del self._partformer_pipeline.scheduler
            if hasattr(self._partformer_pipeline, 'bbox_predictor'):
                del self._partformer_pipeline.bbox_predictor
            del self._partformer_pipeline
            self._partformer_pipeline = None
        
        self._current_model_type = None
        self._clear_vram()
    
    def get_p3sam_model(self) -> AutoMask:
        """Get P3-SAM model, loading it if necessary and unloading other models"""
        with self._lock:
            # If we already have P3-SAM loaded, return it
            if self._current_model_type == "p3sam" and self._p3sam_model is not None:
                return self._p3sam_model
            
            # Unload any currently loaded model
            self._unload_current_model()
            
            # Load P3-SAM model
            print("Loading P3-SAM model...")
            self._p3sam_model = AutoMask()
            
            # Apply quantization to the model
            if hasattr(self._p3sam_model, 'model') and self._p3sam_model.model is not None:
                self._p3sam_model.model = self._apply_quantization(self._p3sam_model.model)
            if hasattr(self._p3sam_model, 'model_parallel') and self._p3sam_model.model_parallel is not None:
                self._p3sam_model.model_parallel = self._apply_quantization(self._p3sam_model.model_parallel)
            
            # Store the target dtype in the AutoMask instance for input conversion
            target_dtype = torch.float16 if self.precision_mode == PrecisionMode.FP8 else self.torch_dtype
            setattr(self._p3sam_model, '_target_dtype', target_dtype)
            
            self._current_model_type = "p3sam"
            print(f"P3-SAM model loaded successfully with {self.precision_mode.value} precision")
            
            return self._p3sam_model
    
    def get_partformer_pipeline(self) -> PartFormerPipeline:
        """Get PartFormer pipeline, loading it if necessary and unloading other models"""
        with self._lock:
            # If we already have PartFormer loaded, return it
            if self._current_model_type == "partformer" and self._partformer_pipeline is not None:
                return self._partformer_pipeline
            
            # Unload any currently loaded model
            self._unload_current_model()
            
            # Load PartFormer pipeline
            print("Loading PartFormer pipeline...")
            import pytorch_lightning as pl
            pl.seed_everything(2026, workers=True)
            
            cfg_path = str(Path(__file__).parent / "XPart/partgen/config" / "infer.yaml")
            config = get_config_from_file(cfg_path)
            assert hasattr(config, "ckpt") or hasattr(
                config, "ckpt_path"
            ), "ckpt or ckpt_path must be specified in config"
            
            self._partformer_pipeline = PartFormerPipeline.from_pretrained(
                model_path="tencent/Hunyuan3D-Part",
                verbose=True,
            )
            
            device = "cuda"
            # Apply quantization to all pipeline components
            if hasattr(self._partformer_pipeline, 'vae') and self._partformer_pipeline.vae is not None:
                self._partformer_pipeline.vae = self._apply_quantization(self._partformer_pipeline.vae)
            if hasattr(self._partformer_pipeline, 'model') and self._partformer_pipeline.model is not None:
                self._partformer_pipeline.model = self._apply_quantization(self._partformer_pipeline.model)
            if hasattr(self._partformer_pipeline, 'conditioner') and self._partformer_pipeline.conditioner is not None:
                self._partformer_pipeline.conditioner = self._apply_quantization(self._partformer_pipeline.conditioner)
            
            # Move to device after quantization
            self._partformer_pipeline.to(device=device, dtype=self.torch_dtype)
            self._current_model_type = "partformer"
            print(f"PartFormer pipeline loaded successfully with {self.precision_mode.value} precision")
            
            return self._partformer_pipeline
    
    def unload_all_models(self):
        """Explicitly unload all models and free VRAM"""
        with self._lock:
            self._unload_current_model()
            print("All models unloaded, VRAM freed")
    
    def set_precision_mode(self, precision_mode: Union[str, PrecisionMode]):
        """Change the precision mode (requires reloading models)"""
        if isinstance(precision_mode, str):
            precision_mode = PrecisionMode(precision_mode.lower())
        
        if precision_mode != self.precision_mode:
            print(f"Changing precision mode from {self.precision_mode.value} to {precision_mode.value}")
            # Unload current models
            self.unload_all_models()
            # Update precision settings
            self.precision_mode = precision_mode
            self.torch_dtype = self._get_torch_dtype()
            # Save to config file
            save_precision_to_config(precision_mode)
            print(f"Precision mode updated. Next model load will use {precision_mode.value}")
    
    def get_memory_usage_estimate(self) -> Dict[str, str]:
        """Get estimated VRAM usage for different precision modes"""
        # These are rough estimates and will vary based on the actual model
        base_vram = {
            "p3sam_fp32": "4-6 GB",
            "partformer_fp32": "4-6 GB",
            "p3sam_fp16": "2-3 GB",
            "partformer_fp16": "2-3 GB",
            "p3sam_bf16": "2-3 GB",
            "partformer_bf16": "2-3 GB",
            "p3sam_fp8": "1-1.5 GB (experimental)",
            "partformer_fp8": "1-1.5 GB (experimental)"
        }
        
        suffix = self.precision_mode.value
        return {
            "current_precision": self.precision_mode.value,
            "p3sam_estimate": base_vram.get(f"p3sam_{suffix}", "Unknown"),
            "partformer_estimate": base_vram.get(f"partformer_{suffix}", "Unknown"),
            "note": "Estimates may vary based on hardware and model complexity"
        }
    
    def get_current_model_type(self) -> Optional[str]:
        """Get the type of currently loaded model"""
        return self._current_model_type


# Global instance with configurable precision
# Precision is loaded from config file if it exists, otherwise defaults to FP16
model_manager = VRAMEfficientModelManager(precision_mode=load_precision_from_config())