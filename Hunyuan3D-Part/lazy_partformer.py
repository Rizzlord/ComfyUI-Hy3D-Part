import torch
import gc
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Any, Union
import warnings

import sys
sys.path.append('XPart')
from partgen.partformer_pipeline import PartFormerPipeline
from partgen.utils.misc import get_config_from_file


class LazyPartFormerPipeline:
    """
    A lazy-loading wrapper for PartFormerPipeline that loads the model only when needed
    and can be explicitly unloaded to free VRAM.
    Supports mixed precision inference for additional VRAM savings.
    """
    
    def __init__(self, model_path="tencent/Hunyuan3D-Part", verbose=True, 
                 dtype: Union[str, torch.dtype] = torch.float32, 
                 enable_mixed_precision: bool = False):
        """
        Initialize the lazy pipeline wrapper
        
        Args:
            model_path: Path to the pre-trained model
            verbose: Whether to print verbose information
            dtype: Data type for model weights (torch.float32, torch.float16, torch.bfloat16)
            enable_mixed_precision: Whether to use mixed precision inference
        """
        self.model_path = model_path
        self.verbose = verbose
        self._pipeline: Optional[Any] = None
        self._is_loaded = False
        self.device = "cuda"
        
        # Handle dtype conversion
        if isinstance(dtype, str):
            dtype_map = {
                "fp32": torch.float32,
                "float32": torch.float32,
                "fp16": torch.float16,
                "float16": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16
            }
            self.dtype = dtype_map.get(dtype.lower(), torch.float32)
        else:
            self.dtype = dtype
            
        self.enable_mixed_precision = enable_mixed_precision
        
        # Validate hardware compatibility
        self._validate_precision_support()
    
    def _validate_precision_support(self):
        """Validate that the selected precision is supported by the hardware"""
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            warnings.warn("BFloat16 not supported on this GPU, falling back to FP16")
            self.dtype = torch.float16
        
        if self.enable_mixed_precision and not torch.cuda.is_available():
            warnings.warn("Mixed precision requires CUDA, disabling mixed precision")
            self.enable_mixed_precision = False
    
    def _apply_precision_optimizations(self, pipeline):
        """Apply precision optimizations to the loaded pipeline"""
        if pipeline is None:
            return pipeline
            
        try:
            # Convert model components to the target dtype
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                pipeline.vae = pipeline.vae.to(dtype=self.dtype)
            if hasattr(pipeline, 'model') and pipeline.model is not None:
                pipeline.model = pipeline.model.to(dtype=self.dtype)
            if hasattr(pipeline, 'conditioner') and pipeline.conditioner is not None:
                pipeline.conditioner = pipeline.conditioner.to(dtype=self.dtype)
                
            print(f"Pipeline components converted to {self.dtype}")
            
            if self.enable_mixed_precision:
                print("Mixed precision inference enabled")
                
        except Exception as e:
            warnings.warn(f"Failed to apply precision optimizations: {e}")
            
        return pipeline
    
    def _load_pipeline(self):
        """Load the PartFormer pipeline if not already loaded"""
        if not self._is_loaded:
            print(f"Loading PartFormer pipeline with {self.dtype} precision...")
            pl.seed_everything(2026, workers=True)
            
            cfg_path = str(Path(__file__).parent / "XPart/partgen/config" / "infer.yaml")
            config = get_config_from_file(cfg_path)
            assert hasattr(config, "ckpt") or hasattr(
                config, "ckpt_path"
            ), "ckpt or ckpt_path must be specified in config"
            
            self._pipeline = PartFormerPipeline.from_pretrained(
                model_path=self.model_path,
                verbose=self.verbose,
            )
            
            # Apply precision optimizations
            self._pipeline = self._apply_precision_optimizations(self._pipeline)
            
            if self._pipeline is not None:
                self._pipeline.to(device=self.device, dtype=self.dtype)
            self._is_loaded = True
            print("PartFormer pipeline loaded successfully")
    
    def _unload_pipeline(self):
        """Unload the PartFormer pipeline and free VRAM"""
        if self._is_loaded and self._pipeline is not None:
            print("Unloading PartFormer pipeline...")
            
            # Move all components to CPU and delete
            if hasattr(self._pipeline, 'vae') and self._pipeline.vae is not None:
                self._pipeline.vae.cpu()
                del self._pipeline.vae
            if hasattr(self._pipeline, 'model') and self._pipeline.model is not None:
                self._pipeline.model.cpu()
                del self._pipeline.model
            if hasattr(self._pipeline, 'conditioner') and self._pipeline.conditioner is not None:
                self._pipeline.conditioner.cpu()
                del self._pipeline.conditioner
            if hasattr(self._pipeline, 'scheduler') and self._pipeline.scheduler is not None:
                del self._pipeline.scheduler
            if hasattr(self._pipeline, 'bbox_predictor') and self._pipeline.bbox_predictor is not None:
                del self._pipeline.bbox_predictor
            
            del self._pipeline
            self._pipeline = None
            
            # Force VRAM cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            self._is_loaded = False
            print("PartFormer pipeline unloaded and VRAM freed")
    
    def __call__(self, *args, **kwargs):
        """
        Execute the pipeline. Load it first if not already loaded.
        Uses mixed precision inference if enabled.
        """
        # Load pipeline if not already loaded
        self._load_pipeline()
        
        try:
            # Use mixed precision autocast if enabled
            if self.enable_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    if self._pipeline is not None:
                        result = self._pipeline(*args, **kwargs)
                        return result
                    else:
                        raise RuntimeError("Pipeline failed to load")
            else:
                # Standard inference
                if self._pipeline is not None:
                    result = self._pipeline(*args, **kwargs)
                    return result
                else:
                    raise RuntimeError("Pipeline failed to load")
        finally:
            # Optionally unload pipeline after use to free VRAM
            # Comment out the next line if you want to keep the pipeline loaded for multiple calls
            self._unload_pipeline()
    
    def to(self, device=None, dtype=None):
        """Set device and dtype for future loading"""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        
        # If pipeline is already loaded, move it to the new device/dtype
        if self._is_loaded and self._pipeline is not None:
            self._pipeline.to(device=device, dtype=dtype)
    
    def unload(self):
        """Explicitly unload the pipeline"""
        self._unload_pipeline()
    
    @property
    def is_loaded(self):
        """Check if the pipeline is currently loaded"""
        return self._is_loaded
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying pipeline.
        Load the pipeline if it's not already loaded and the attribute is accessed.
        """
        if name.startswith('_') or name in ['model_path', 'verbose', 'device', 'dtype']:
            # Don't load pipeline for private attributes or our own attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Load pipeline if not already loaded
        self._load_pipeline()
        
        # Return the attribute from the loaded pipeline
        return getattr(self._pipeline, name)