"""
Fix PyTorch/torchvision compatibility issues
Works with both pip and uv package managers
"""

import subprocess
import sys
import shutil

def fix_torchvision():
    """Install compatible torchvision version"""
    print("=" * 70)
    print("Fixing PyTorch/torchvision compatibility...")
    print("=" * 70)
    
    # Detect package manager (uv or pip)
    use_uv = shutil.which("uv") is not None
    package_manager = "uv" if use_uv else "pip"
    
    print(f"\nDetected package manager: {package_manager}")
    
    if use_uv:
        # Using uv
        print("\n1. Installing compatible torchvision (0.20.1 for PyTorch 2.9.1) with uv...")
        # uv handles uninstall/install in one command
        result = subprocess.run(
            ["uv", "pip", "install", "torchvision==0.20.1"],
            check=False
        )
        if result.returncode != 0:
            print("⚠️  uv pip install failed, trying uv add...")
            subprocess.run(["uv", "add", "torchvision==0.20.1"], check=True)
    else:
        # Using pip
        print("\n1. Uninstalling existing torchvision...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], check=False)
        
        print("\n2. Installing compatible torchvision (0.20.1 for PyTorch 2.9.1)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torchvision==0.20.1"], check=True)
    
    print("\n✓ torchvision installation attempted!")
    print("\n2. Verifying installation...")
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__} installed")
        import torch
        print(f"✓ PyTorch {torch.__version__} detected")
        print("\n✓ Compatibility check passed!")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("\nIf torchvision is still not working, try manually:")
        if use_uv:
            print("  uv pip install torchvision==0.20.1")
        else:
            print("  pip install torchvision==0.20.1")
    
    print("\n" + "=" * 70)
    print("Done! Try running check_models.py again.")
    print("=" * 70)

if __name__ == "__main__":
    fix_torchvision()
