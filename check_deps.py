#!/usr/bin/env python3
"""
Simple dependency checker for Hunyuan3D-Part
"""
import subprocess
import sys

def check_deps():
    # Get installed packages
    result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                          capture_output=True, text=True)
    installed_packages = result.stdout.lower()
    
    # Required packages
    required = [
        "torch", "numpy", "trimesh", "scikit-learn", "fpsample",
        "pytorch-lightning", "diffusers", "addict", "numba",
        "spconv", "omegaconf"
    ]
    
    missing = []
    for pkg in required:
        pkg_check = pkg.replace("_", "-").lower()
        if pkg_check not in installed_packages:
            missing.append(pkg)
        else:
            print(f"✓ {pkg}")
    
    if missing:
        print(f"\n✗ Missing: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
    else:
        print("\n🎉 All dependencies found!")

if __name__ == "__main__":
    check_deps()