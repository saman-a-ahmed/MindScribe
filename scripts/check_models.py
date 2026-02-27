"""
Check if models are available and can be loaded
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import JournalAnalyzer
from app.config import settings


def check_model_path(path: str, name: str) -> bool:
    """Check if model path exists"""
    if os.path.exists(path):
        print(f"✓ {name} model found at: {path}")
        return True
    else:
        print(f"✗ {name} model NOT found at: {path}")
        return False


def main():
    """Check model availability"""
    print("=" * 70)
    print("Model Availability Check")
    print("=" * 70)
    print()
    
    # Check paths
    emotion_ok = check_model_path(settings.EMOTION_MODEL_PATH, "Emotion")
    distortion_ok = check_model_path(settings.DISTORTION_MODEL_PATH, "Distortion")
    
    print()
    print("=" * 70)
    print("Attempting to load models...")
    print("=" * 70)
    print()
    
    try:
        analyzer = JournalAnalyzer(
            emotion_model_path=settings.EMOTION_MODEL_PATH,
            distortion_model_path=settings.DISTORTION_MODEL_PATH,
            verbose=True
        )
        
        status = analyzer.get_status()
        print()
        print("=" * 70)
        print("Model Status:")
        print("=" * 70)
        print(f"Emotion Classifier: {'✓ Loaded' if status['emotion_classifier'] else '✗ Failed'}")
        print(f"Distortion Classifier: {'✓ Loaded' if status['distortion_classifier'] else '✗ Failed'}")
        print(f"Ready: {'✓ Yes' if status['ready'] else '✗ No'}")
        print()
        
        if status['ready']:
            print("✓ All available models loaded successfully!")
            return 0
        else:
            print("⚠️  Some models failed to load. Check paths and model files.")
            return 1
    
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
