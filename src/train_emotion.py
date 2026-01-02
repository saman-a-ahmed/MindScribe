"""
Training script for multi-label emotion classification
Uses RoBERTa-base with BCEWithLogitsLoss
"""

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os
import logging
logging.basicConfig(level=logging.INFO)

# Emotion labels for go_emotions (for interpretation)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


def compute_metrics(eval_pred):
    """
    Compute metrics for multi-label classification
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        Dictionary of metric scores
    """
    predictions, labels = eval_pred
    
    # Apply sigmoid to logits to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    
    # Convert probabilities to binary predictions (threshold = 0.3)
    y_pred = (probs.numpy() > 0.3).astype(int)
    y_true = labels
    
    # Calculate metrics
    # Note: 'samples' average computes metrics for each instance and averages
    metrics = {
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'accuracy_subset': accuracy_score(y_true, y_pred),  # Exact match accuracy
    }
    
    return metrics


def load_data(data_dir="data/processed"):
    """
    Load preprocessed datasets
    
    Args:
        data_dir: Directory containing preprocessed data
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    print(f"\n📂 Loading preprocessed data from {data_dir}...")
    
    train_dataset = load_from_disk(f"{data_dir}/train")
    val_dataset = load_from_disk(f"{data_dir}/val")
    test_dataset = load_from_disk(f"{data_dir}/test")
    
    print(f"✓ Data loaded:")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")
    
    return train_dataset, val_dataset, test_dataset


def load_model(model_name="roberta-base", num_labels=28):
    """
    Load pre-trained model for multi-label classification
    
    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of emotion labels
    
    Returns:
        model, tokenizer
    """
    print(f"\n🤖 Loading model: {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",  # KEY: Uses BCEWithLogitsLoss
        ignore_mismatched_sizes=True  # Ignore size mismatch for classification head
    )
    
    print(f"✓ Model loaded:")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Labels: {num_labels}")
    print(f"  Problem type: multi_label_classification")
    
    return model, tokenizer


def create_trainer(
    model,
    train_dataset,
    val_dataset,
    output_dir="models/emotion_classifier",
    num_epochs=10,
    batch_size=16,
    learning_rate=1e-5
):
    """
    Create HuggingFace Trainer with training arguments
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
    
    Returns:
        Trainer object
    """
    print(f"\n⚙️  Setting up training...")
    
    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training hyperparameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,              # L2 regularization
        
        # Learning rate schedule
        warmup_steps=500,                # Gradual warmup
        lr_scheduler_type="linear",      # Linear decay
        
        # Evaluation
        eval_strategy="steps",           # Evaluate every N steps
        eval_steps=500,                  # Evaluate every 500 steps
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,              # Keep only 2 best checkpoints
        load_best_model_at_end=True,     # Load best model after training
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        report_to="none",               
        
        # Performance
        fp16=torch.cuda.is_available(),  
        dataloader_num_workers=4,
        
        # Reproducibility
        seed=42,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )
    
    print(f"✓ Trainer configured:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    return trainer


def predict_sample(model, tokenizer, text, threshold=0.3):
    """
    Predict emotions for a single text
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Input text
        threshold: Prediction threshold
    
    Returns:
        List of predicted emotion labels
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply sigmoid and threshold
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    predictions = (probs > threshold).astype(int)
    
    # Get predicted emotion labels
    predicted_emotions = [
        (EMOTION_LABELS[i], probs[i]) 
        for i, pred in enumerate(predictions) if pred == 1
    ]
    
    return predicted_emotions, probs


def test_model_predictions(model, tokenizer):
    """
    Test model on sample texts to verify it's working
    """
    print("\n🧪 Testing model on sample texts...")
    print("=" * 70)
    
    test_texts = [
        "I just got promoted! This is the best day ever!",
        "I can't believe they cancelled my favorite show. So disappointed.",
        "This traffic is driving me crazy!",
        "I'm really worried about the exam tomorrow.",
    ]
    
    for text in test_texts:
        emotions, probs = predict_sample(model, tokenizer, text)
        print(f"\nText: {text}")
        print(f"Predicted emotions:")
        for emotion, prob in emotions:
            print(f"  - {emotion}: {prob:.3f}")
        if not emotions:
            print("  - neutral (no strong emotions detected)")
    
    print("=" * 70)


def train_model(
    data_dir="data/processed",
    output_dir="models/emotion_classifier",
    num_epochs=5,
    batch_size=16
):
    """
    Main training pipeline
    
    Args:
        data_dir: Directory with preprocessed data
        output_dir: Directory to save trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
    """
    
    print("\n" + "="*70)
    print("🚀 EMOTION CLASSIFICATION TRAINING")
    print("="*70)
    
    # Step 1: Load data
    train_dataset, val_dataset, test_dataset = load_data(data_dir)
    
    # Step 2: Load model
    model, tokenizer = load_model(num_labels=28)
    
    # Step 3: Create trainer
    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Step 4: Train
    print("\n🏋️  Starting training...")
    print("-" * 70)
    train_result = trainer.train()
    
    # Step 5: Save model
    print("\n💾 Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to {output_dir}")
    
    # Step 6: Evaluate on validation set
    print("\n📊 Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print("\nValidation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Step 7: Test predictions
    test_model_predictions(model, tokenizer)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {output_dir}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"Best validation F1 (micro): {eval_results['eval_f1_micro']:.4f}")
    
    return trainer, model, tokenizer


if __name__ == "__main__":
    """
    Run training
    Usage: python src/train_emotion.py
    """
    
    # Check if preprocessed data exists
    if not os.path.exists("data/processed/train"):
        print("❌ Preprocessed data not found!")
        print("Please run preprocessing first:")
        print("  python src/preprocessing.py")
        exit(1)
    
    # Train model
    trainer, model, tokenizer = train_model(
        data_dir="data/processed",
        output_dir="models/emotion_classifier",
        num_epochs=10,
        batch_size=16  # Adjust if GPU memory is limited
    )
