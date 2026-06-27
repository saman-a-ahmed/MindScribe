"""
Inference script for cognitive distortion classification
Run sanity checks on journal-like texts
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from src.cognitive_distortions import (
    COGNITIVE_DISTORTION_LABELS,
    DISTORTION_NAMES,
    DISTORTION_DESCRIPTIONS,
    get_num_distortions
)


class DistortionClassifier:
    """
    Wrapper for trained cognitive distortion classification model
    """
    
    def __init__(self, model_path="models/distortion_classifier"):
        """
        Load trained model and tokenizer
        
        Args:
            model_path: Path to saved model directory
        """
        print(f"Loading model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        num_labels = get_num_distortions()
        # Load model using AutoModel - it will infer from model_type in config
        # Use local_files_only to avoid network calls and trust_remote_code=False for security
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                trust_remote_code=False,
                local_files_only=True
            )
        except Exception as e:
            # If loading fails, provide detailed error
            error_msg = str(e)
            print(f"Error loading model: {error_msg}")
            # Try without local_files_only in case there are missing files
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=num_labels,
                    trust_remote_code=False
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to load model from {model_path}. First error: {error_msg}. Second error: {str(e2)}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Detecting {num_labels} cognitive distortion types")
    
    def predict(self, text, threshold=0.3, top_k=None):
        """
        Predict cognitive distortions for a given text
        
        Args:
            text: Input text string
            threshold: Probability threshold for binary predictions
            top_k: Number of top distortions to return (None = all above threshold)
        
        Returns:
            dict with 'detected_distortions', 'top_distortions', 'all_probabilities'
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        # Get all distortions above threshold
        binary_preds = (probs > threshold).astype(int)
        detected_distortions = []
        for i, pred in enumerate(binary_preds):
            if pred == 1:
                detected_distortions.append({
                    'label': COGNITIVE_DISTORTION_LABELS[i],
                    'name': DISTORTION_NAMES[COGNITIVE_DISTORTION_LABELS[i]],
                    'description': DISTORTION_DESCRIPTIONS[COGNITIVE_DISTORTION_LABELS[i]],
                    'probability': float(probs[i])
                })
        
        # Sort by probability (descending)
        detected_distortions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Get top-k distortions regardless of threshold
        if top_k is None:
            top_k = len(COGNITIVE_DISTORTION_LABELS)
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_distortions = []
        for i in top_indices:
            top_distortions.append({
                'label': COGNITIVE_DISTORTION_LABELS[i],
                'name': DISTORTION_NAMES[COGNITIVE_DISTORTION_LABELS[i]],
                'description': DISTORTION_DESCRIPTIONS[COGNITIVE_DISTORTION_LABELS[i]],
                'probability': float(probs[i])
            })
        
        return {
            'detected_distortions': detected_distortions,
            'top_distortions': top_distortions,
            'all_probabilities': probs.tolist(),
            'has_distortions': len(detected_distortions) > 0
        }
    
    def print_prediction(self, text, threshold=0.3, top_k=5):
        """
        Predict and print results in a readable format
        
        Args:
            text: Input text
            threshold: Probability threshold
            top_k: Number of top distortions to show
        """
        print("\n" + "="*70)
        print("📝 INPUT TEXT:")
        print(f"   {text}")
        print("="*70)
        
        result = self.predict(text, threshold, top_k)
        
        print(f"\n🎯 TOP {top_k} PREDICTED DISTORTIONS:")
        print("-"*70)
        for i, dist_data in enumerate(result['top_distortions'], 1):
            name = dist_data['name']
            prob = dist_data['probability']
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"{i}. {name:30s} {prob:.4f} {bar}")
        
        if result['detected_distortions']:
            print(f"\n✓ DISTORTIONS DETECTED (above threshold {threshold}):")
            print("-"*70)
            for i, dist_data in enumerate(result['detected_distortions'], 1):
                print(f"\n  {i}. {dist_data['name']} ({dist_data['label']})")
                print(f"     Probability: {dist_data['probability']:.4f}")
                print(f"     Description: {dist_data['description']}")
        else:
            print(f"\n✓ NO DISTORTIONS DETECTED (above threshold {threshold})")
            print("   This might indicate healthy/balanced thinking")
        
        print("="*70 + "\n")


def run_sanity_checks(model_path="models/distortion_classifier"):
    """
    Run sanity checks on various journal-like texts
    
    Args:
        model_path: Path to trained model
    """
    # Initialize classifier
    classifier = DistortionClassifier(model_path)
    
    # Test cases: journal-like entries with various distortions
    test_texts = [
        # Fortune telling + catastrophizing
        "I'm definitely going to fail this presentation tomorrow. My entire career will be ruined.",
        
        # All-or-nothing + labeling
        "If I'm not perfect, I'm a total failure. I'm just a loser.",
        
        # Mind reading + emotional reasoning
        "They probably think I'm stupid. I feel anxious, so something bad must be about to happen.",
        
        # Should statements
        "I should be able to handle this. I shouldn't make any mistakes. I must be perfect.",
        
        # Personalization
        "My team lost because I wasn't supportive enough. It's all my fault.",
        
        # Overgeneralization
        "I failed this test, so I'm bad at everything. I always mess things up.",
        
        # Mental filter
        "I got 9 compliments today but I only remember the one criticism. The whole day was terrible.",
        
        # Neutral/balanced
        "Today was actually pretty good. I got some work done, made progress on my project, and had a nice conversation with a colleague.",
        
        # Discounting positive
        "They only complimented me to be nice. Anyone could have done that. It doesn't really count.",
    ]
    
    print("\n" + "🧪 RUNNING SANITY CHECKS ON COGNITIVE DISTORTION MODEL" + "\n")
    print("Testing model on journal-like texts to verify predictions make sense...")
    print("\n")
    
    # Run predictions on all test cases
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE #{i}")
        classifier.print_prediction(text, threshold=0.3, top_k=5)
        
        if i < len(test_texts):
            input("Press Enter to continue to next test case...")
    
    print("\n" + "="*70)
    print("✅ SANITY CHECK COMPLETE")
    print("="*70)
    print("\n")


def interactive_mode(model_path="models/distortion_classifier"):
    """
    Interactive mode: type your own texts
    
    Args:
        model_path: Path to trained model
    """
    classifier = DistortionClassifier(model_path)
    
    print("\n" + "="*70)
    print("🎮 INTERACTIVE COGNITIVE DISTORTION PREDICTION MODE")
    print("="*70)
    print("\nType your journal entry or any text to analyze for cognitive distortions.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        text = input("📝 Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not text:
            print("⚠️  Please enter some text.\n")
            continue
        
        classifier.print_prediction(text, threshold=0.3, top_k=5)


def analyze_journal_entry(text, model_path="models/distortion_classifier", threshold=0.3):
    """
    Analyze a journal entry and return structured results
    
    Args:
        text: Journal entry text
        model_path: Path to trained model
        threshold: Probability threshold
        
    Returns:
        dict with analysis results
    """
    classifier = DistortionClassifier(model_path)
    result = classifier.predict(text, threshold=threshold)
    
    return {
        'text': text,
        'has_distortions': result['has_distortions'],
        'detected_distortions': result['detected_distortions'],
        'top_distortions': result['top_distortions'][:5],  # Top 5
        'summary': {
            'count': len(result['detected_distortions']),
            'distortions': [d['name'] for d in result['detected_distortions']]
        }
    }


if __name__ == "__main__":
    """
    Run inference sanity checks. Run from the project root as a module:
      python -m src.inference_distortion                # Run predefined test cases
      python -m src.inference_distortion --interactive  # Interactive mode
    """
    import sys
    import os

    # Check if model exists
    if not os.path.exists("models/distortion_classifier"):
        print("❌ Model not found!")
        print("Please train the model first:")
        print("  python -m src.train_distortion")
        sys.exit(1)
    
    # Check for interactive flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--interactive', '-i']:
        interactive_mode()
    else:
        run_sanity_checks()
