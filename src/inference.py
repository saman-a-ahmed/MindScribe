"""
Inference script for emotion classification
Run sanity checks on journal-like texts
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np


# Emotion labels for go_emotions
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


class EmotionClassifier:
    """
    Wrapper for trained emotion classification model
    """
    
    def __init__(self, model_path="models/emotion_classifier"):
        """
        Load trained model and tokenizer
        
        Args:
            model_path: Path to saved model directory
        """
        print(f"Loading model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model using AutoModel - it will infer from model_type in config
        # Use local_files_only to avoid network calls and trust_remote_code=False for security
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
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
                    trust_remote_code=False
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to load model from {model_path}. First error: {error_msg}. Second error: {str(e2)}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def predict(self, text, threshold=0.5, top_k=5):
        """
        Predict emotions for a given text
        
        Args:
            text: Input text string
            threshold: Probability threshold for binary predictions
            top_k: Number of top emotions to return
        
        Returns:
            dict with 'all_emotions', 'top_emotions', 'probabilities'
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
        
        # Get all emotions above threshold
        binary_preds = (probs > threshold).astype(int)
        detected_emotions = [
            {'emotion': EMOTION_LABELS[i], 'probability': float(probs[i])}
            for i, pred in enumerate(binary_preds) if pred == 1
        ]
        
        # Get top-k emotions regardless of threshold
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_emotions = [
            {'emotion': EMOTION_LABELS[i], 'probability': float(probs[i])}
            for i in top_indices
        ]
        
        return {
            'all_emotions': detected_emotions,
            'top_emotions': top_emotions,
            'probabilities': probs
        }
    
    def print_prediction(self, text, threshold=0.5, top_k=5):
        """
        Predict and print results in a readable format
        
        Args:
            text: Input text
            threshold: Probability threshold
            top_k: Number of top emotions to show
        """
        print("\n" + "="*70)
        print("📝 INPUT TEXT:")
        print(f"   {text}")
        print("="*70)
        
        result = self.predict(text, threshold, top_k)
        
        print(f"\n🎯 TOP {top_k} PREDICTED EMOTIONS:")
        print("-"*70)
        for i, emotion_data in enumerate(result['top_emotions'], 1):
            emotion = emotion_data['emotion']
            prob = emotion_data['probability']
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"{i}. {emotion:15s} {prob:.4f} {bar}")
        
        if result['all_emotions']:
            print(f"\n✓ EMOTIONS ABOVE THRESHOLD ({threshold}):")
            print("-"*70)
            for emotion_data in result['all_emotions']:
                print(f"  • {emotion_data['emotion']:15s} ({emotion_data['probability']:.4f})")
        else:
            print(f"\n⚠️  No emotions detected above threshold ({threshold})")
            print("   (This might indicate 'neutral' or very low confidence)")
        
        print("="*70 + "\n")


def run_sanity_checks(model_path="models/emotion_classifier"):
    """
    Run sanity checks on various journal-like texts
    
    Args:
        model_path: Path to trained model
    """
    # Initialize classifier
    classifier = EmotionClassifier(model_path)
    
    # Test cases: journal-like entries
    test_texts = [
        # Original test case
        "I feel exhausted lately, even small tasks feel overwhelming.",
        
        # Additional diverse test cases
        "I just got the promotion I've been working towards for years! Can't believe it's finally happening!",
        
        "My dog passed away yesterday. The house feels so empty without him.",
        
        "I don't understand why they would say something like that to me. It doesn't make any sense.",
        
        "Today was just... a day. Nothing special happened. Just went through the motions.",
        
        "I'm really worried about the presentation tomorrow. What if I mess up in front of everyone?",
        
        "Someone left a kind note on my desk today. It made my whole week better.",
        
        "I can't stand how they keep interrupting me in meetings. It's so disrespectful!",
    ]
    
    print("\n" + "🧪 RUNNING SANITY CHECKS ON MODEL" + "\n")
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


def interactive_mode(model_path="models/emotion_classifier"):
    """
    Interactive mode: type your own texts
    
    Args:
        model_path: Path to trained model
    """
    classifier = EmotionClassifier(model_path)
    
    print("\n" + "="*70)
    print("🎮 INTERACTIVE EMOTION PREDICTION MODE")
    print("="*70)
    print("\nType your journal entry or any text to analyze.")
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


if __name__ == "__main__":
    """
    Run inference sanity checks. Run from the project root as a module:
      python -m src.inference                # Run predefined test cases
      python -m src.inference --interactive  # Interactive mode
    """
    import sys
    import os

    # Check if model exists
    if not os.path.exists("models/emotion_classifier"):
        print("❌ Model not found!")
        print("Please train the model first:")
        print("  python -m src.train_emotion")
        sys.exit(1)
    
    # Check for interactive flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--interactive', '-i']:
        interactive_mode()
    else:
        run_sanity_checks()