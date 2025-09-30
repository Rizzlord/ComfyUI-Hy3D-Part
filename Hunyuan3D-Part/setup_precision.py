#!/usr/bin/env python3
"""
Precision Configuration Setup Script

This script helps you configure the optimal precision mode for your hardware
and use case. Run this script to automatically detect and configure the best
settings for your system.

Usage:
    python setup_precision.py [--interactive] [--precision fp16|bf16|fp32]
"""

import argparse
import torch
import sys
from pathlib import Path

# Add the local modules to path
sys.path.append(str(Path(__file__).parent))

try:
    from model_manager import model_manager, PrecisionMode, save_precision_to_config
except ImportError:
    print("Error: Could not import model_manager. Make sure you're in the correct directory.")
    sys.exit(1)


def check_hardware_capabilities():
    """Check what precision modes are supported by the current hardware"""
    capabilities = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': None,
        'total_vram_gb': None,
        'supports_fp16': False,
        'supports_bf16': False,
        'supports_fp8': False
    }
    
    if torch.cuda.is_available():
        capabilities['gpu_name'] = torch.cuda.get_device_name(0)
        capabilities['total_vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        capabilities['supports_fp16'] = True  # Most modern GPUs support FP16
        capabilities['supports_bf16'] = torch.cuda.is_bf16_supported()
        capabilities['supports_fp8'] = hasattr(torch, 'float8_e4m3fn')  # Very limited support
    
    return capabilities


def recommend_precision(capabilities, use_case="balanced"):
    """Recommend optimal precision based on hardware and use case"""
    
    if not capabilities['cuda_available']:
        return PrecisionMode.FP32, "CUDA not available, using FP32"
    
    vram = capabilities['total_vram_gb']
    gpu_name = capabilities['gpu_name']
    
    # Use case specific recommendations
    if use_case == "quality":
        if vram >= 12:
            return PrecisionMode.FP32, "High VRAM + quality focus: FP32"
        elif capabilities['supports_bf16']:
            return PrecisionMode.BF16, "Medium VRAM + quality focus: BF16"
        else:
            return PrecisionMode.FP16, "Limited options: FP16"
    
    elif use_case == "memory":
        if capabilities['supports_bf16']:
            return PrecisionMode.BF16, "Memory efficient + good quality: BF16"
        else:
            return PrecisionMode.FP16, "Memory efficient: FP16"
    
    else:  # balanced
        if vram >= 12 and capabilities['supports_bf16']:
            return PrecisionMode.BF16, "High VRAM + modern GPU: BF16"
        elif vram >= 8:
            if capabilities['supports_bf16']:
                return PrecisionMode.BF16, "Medium VRAM + modern GPU: BF16"
            else:
                return PrecisionMode.FP16, "Medium VRAM: FP16"
        else:
            return PrecisionMode.FP16, "Low VRAM: FP16"


def print_system_info(capabilities):
    """Print detailed system information"""
    print("=== System Information ===")
    print(f"CUDA Available: {capabilities['cuda_available']}")
    
    if capabilities['cuda_available']:
        print(f"GPU: {capabilities['gpu_name']}")
        print(f"Total VRAM: {capabilities['total_vram_gb']:.1f} GB")
        print(f"FP16 Support: {capabilities['supports_fp16']}")
        print(f"BF16 Support: {capabilities['supports_bf16']}")
        print(f"FP8 Support: {capabilities['supports_fp8']} (experimental)")
    else:
        print("GPU: None (CPU only)")
    
    print()


def print_precision_comparison():
    """Print comparison table of different precision modes"""
    print("=== Precision Mode Comparison ===")
    print("| Mode | VRAM Usage | Quality | Speed | Compatibility |")
    print("|------|------------|---------|-------|---------------|")
    print("| FP32 | 100%       | Best    | 1.0x  | All GPUs      |")
    print("| BF16 | ~50%       | Excellent| 1.2x  | RTX 30xx+     |")
    print("| FP16 | ~50%       | Very Good| 1.3x  | Most GPUs     |")
    print("| FP8  | ~25%       | Good    | 1.8x  | Very Limited  |")
    print()


def interactive_setup():
    """Interactive setup wizard"""
    print("🔧 Hunyuan3D-Part Precision Configuration Wizard")
    print("=" * 50)
    
    # Check hardware
    capabilities = check_hardware_capabilities()
    print_system_info(capabilities)
    
    if not capabilities['cuda_available']:
        print("⚠️  No CUDA GPU detected. You'll be limited to CPU inference with FP32.")
        model_manager.set_precision_mode(PrecisionMode.FP32)
        return PrecisionMode.FP32
    
    print_precision_comparison()
    
    # Ask about use case
    print("What's your primary use case?")
    print("1. Quality (research, high-quality outputs)")
    print("2. Memory Efficiency (limited VRAM)")
    print("3. Balanced (good quality + efficiency)")
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice == "1":
            use_case = "quality"
            break
        elif choice == "2":
            use_case = "memory"
            break
        elif choice == "3":
            use_case = "balanced"
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Get recommendation
    recommended_precision, reason = recommend_precision(capabilities, use_case)
    
    print(f"\n✅ Recommendation: {recommended_precision.value.upper()}")
    print(f"   Reason: {reason}")
    
    # Ask for confirmation
    confirm = input(f"\nApply {recommended_precision.value.upper()} precision? (y/n): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        model_manager.set_precision_mode(recommended_precision)
        save_precision_to_config(recommended_precision)
        print(f"✅ Precision set to {recommended_precision.value.upper()}")
        
        # Show memory estimates
        usage = model_manager.get_memory_usage_estimate()
        print(f"\n📊 Estimated VRAM Usage:")
        print(f"   P3-SAM: {usage['p3sam_estimate']}")
        print(f"   PartFormer: {usage['partformer_estimate']}")
        
        return recommended_precision
    else:
        print("❌ Configuration cancelled. Using default FP32.")
        return PrecisionMode.FP32


def auto_setup(precision_mode=None):
    """Automatic setup with optional precision override"""
    capabilities = check_hardware_capabilities()
    
    if precision_mode:
        try:
            precision = PrecisionMode(precision_mode.lower())
            
            # Check compatibility
            if precision == PrecisionMode.BF16 and not capabilities['supports_bf16']:
                print(f"⚠️  BF16 not supported on {capabilities['gpu_name']}, falling back to FP16")
                precision = PrecisionMode.FP16
            
            model_manager.set_precision_mode(precision)
            save_precision_to_config(precision)
            print(f"✅ Precision manually set to {precision.value.upper()}")
            return precision
            
        except ValueError:
            print(f"❌ Invalid precision mode: {precision_mode}")
            print("Valid options: fp32, fp16, bf16, fp8")
            return None
    
    else:
        # Auto-recommend
        recommended_precision, reason = recommend_precision(capabilities, "balanced")
        model_manager.set_precision_mode(recommended_precision)
        save_precision_to_config(recommended_precision)
        print(f"✅ Auto-configured precision: {recommended_precision.value.upper()}")
        print(f"   Reason: {reason}")
        return recommended_precision


def main():
    parser = argparse.ArgumentParser(description="Configure optimal precision for Hunyuan3D-Part")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run interactive configuration wizard")
    parser.add_argument("--precision", "-p", choices=["fp32", "fp16", "bf16", "fp8"],
                       help="Set specific precision mode")
    parser.add_argument("--info", action="store_true",
                       help="Show system information and current settings")
    
    args = parser.parse_args()
    
    if args.info:
        capabilities = check_hardware_capabilities()
        print_system_info(capabilities)
        current_precision = model_manager.precision_mode
        print(f"Current Precision: {current_precision.value.upper()}")
        
        usage = model_manager.get_memory_usage_estimate()
        print(f"\nEstimated VRAM Usage:")
        print(f"P3-SAM: {usage['p3sam_estimate']}")
        print(f"PartFormer: {usage['partformer_estimate']}")
        return
    
    if args.interactive:
        interactive_setup()
    else:
        auto_setup(args.precision)
    
    print("\n🎉 Configuration complete!")
    print("You can now run the Hunyuan3D-Part application with optimized precision.")
    print("\nTo change precision later, run:")
    print("  python setup_precision.py --precision fp16")
    print("  python setup_precision.py --interactive")


if __name__ == "__main__":
    main()