#!/usr/bin/env python3
"""
Test script for ComfyUI-Hy3D-Part integration

This script tests the basic functionality of the Hunyuan3D-Part nodes
without requiring a full ComfyUI installation.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test if the nodes can be imported successfully."""
    print("Testing imports...")
    
    try:
        from hy3dpart_nodes import (
            Hy3DPartSegmentation,
            Hy3DPartGeneration, 
            Hy3DPartPipeline,
            Hy3DExportParts,
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS
        )
        print("✓ Node imports successful")
        
        # Check node mappings
        assert len(NODE_CLASS_MAPPINGS) == 4, "Expected 4 node classes"
        assert len(NODE_DISPLAY_NAME_MAPPINGS) == 4, "Expected 4 display names"
        print("✓ Node mappings are correct")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_node_structure():
    """Test node input/output type definitions."""
    print("\nTesting node structure...")
    
    try:
        from hy3dpart_nodes import Hy3DPartSegmentation, Hy3DPartGeneration
        
        # Test Segmentation node
        seg_inputs = Hy3DPartSegmentation.INPUT_TYPES()
        assert "required" in seg_inputs
        assert "trimesh" in seg_inputs["required"]
        print("✓ Segmentation node structure is valid")
        
        # Test Generation node
        gen_inputs = Hy3DPartGeneration.INPUT_TYPES()
        assert "required" in gen_inputs
        assert "trimesh" in gen_inputs["required"]
        print("✓ Generation node structure is valid")
        
        return True
        
    except Exception as e:
        print(f"✗ Node structure test failed: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available."""
    print("\nTesting dependencies...")
    
    dependencies = {
        "torch": "PyTorch",
        "numpy": "NumPy", 
        "trimesh": "Trimesh",
        "tempfile": "Tempfile (built-in)",
        "pathlib": "Pathlib (built-in)",
    }
    
    missing = []
    for dep, name in dependencies.items():
        try:
            __import__(dep)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} missing")
            missing.append(name)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install torch numpy trimesh")
        return False
    
    return True

def test_hunyuan_structure():
    """Test if Hunyuan3D-Part directory structure is correct."""
    print("\nTesting Hunyuan3D-Part structure...")
    
    hunyuan_dir = current_dir / "Hunyuan3D-Part"
    if not hunyuan_dir.exists():
        print("✗ Hunyuan3D-Part directory not found")
        print("Please clone Hunyuan3D-Part repository:")
        print("git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git")
        return False
    
    required_paths = [
        hunyuan_dir / "P3-SAM",
        hunyuan_dir / "P3-SAM" / "model.py",
        hunyuan_dir / "P3-SAM" / "demo" / "auto_mask.py",
        hunyuan_dir / "XPart",
        hunyuan_dir / "XPart" / "partgen",
        hunyuan_dir / "XPart" / "partgen" / "partformer_pipeline.py",
        hunyuan_dir / "XPart" / "partgen" / "config" / "infer.yaml",
    ]
    
    missing = []
    for path in required_paths:
        if path.exists():
            print(f"✓ {path.relative_to(hunyuan_dir)}")
        else:
            print(f"✗ {path.relative_to(hunyuan_dir)} missing")
            missing.append(str(path.relative_to(hunyuan_dir)))
    
    if missing:
        print(f"\nMissing files/directories: {', '.join(missing)}")
        return False
    
    return True

def create_test_mesh():
    """Create a simple test mesh for testing."""
    try:
        import trimesh
        import numpy as np
        
        # Create a simple cube mesh
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
        ])
        
        faces = np.array([
            [0, 1, 2], [2, 3, 0],  # bottom
            [4, 7, 6], [6, 5, 4],  # top
            [0, 4, 5], [5, 1, 0],  # front
            [2, 6, 7], [7, 3, 2],  # back
            [0, 3, 7], [7, 4, 0],  # left
            [1, 5, 6], [6, 2, 1]   # right
        ])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print("✓ Test mesh created successfully")
        return mesh
        
    except Exception as e:
        print(f"✗ Failed to create test mesh: {e}")
        return None

def test_basic_functionality():
    """Test basic node functionality without actual processing."""
    print("\nTesting basic functionality...")
    
    try:
        from hy3dpart_nodes import Hy3DPartSegmentation, HUNYUAN_AVAILABLE
        
        # Create test mesh
        test_mesh = create_test_mesh()
        if test_mesh is None:
            return False
        
        # Initialize segmentation node
        seg_node = Hy3DPartSegmentation()
        
        if not HUNYUAN_AVAILABLE:
            print("⚠ Hunyuan3D-Part modules not available - this is expected if not fully installed")
            print("✓ Node initialization successful (without Hunyuan modules)")
            return True
        
        print("✓ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ComfyUI-Hy3D-Part Integration Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Node Structure Test", test_node_structure), 
        ("Dependencies Test", test_dependencies),
        ("Hunyuan Structure Test", test_hunyuan_structure),
        ("Basic Functionality Test", test_basic_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! ComfyUI-Hy3D-Part integration is ready.")
    else:
        print("\n⚠ Some tests failed. Please check the requirements and installation.")
        
    print("\nNext steps:")
    print("1. Install missing dependencies if any")
    print("2. Clone Hunyuan3D-Part repository if not present")
    print("3. Install Sonata and other Hunyuan3D-Part dependencies")
    print("4. Test in ComfyUI with actual mesh files")

if __name__ == "__main__":
    main()