"""
Training script for multi-label cognitive distortion classification
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
from cognitive_distortions import (
    COGNITIVE_DISTORTION_LABELS,
    get_num_distortions,
    DISTORTION_NAMES
)
from torch.nn import BCEWithLogitsLoss
logging.basicConfig(level=logging.INFO)


def find_optimal_threshold(probs, labels, metric='f1'):
    """
    Find optimal threshold for multi-label classification
    
    Args:
        probs: Probability predictions (n_samples, n_classes)
        labels: True labels (n_samples, n_classes)
        metric: Metric to optimize ('f1', 'f1_micro', 'f1_macro')
    
    Returns:
        Optimal threshold value
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (probs > threshold).astype(int)
        
        if metric == 'f1_micro':
            score = f1_score(labels, y_pred, average='micro', zero_division=0)
        elif metric == 'f1_macro':
            score = f1_score(labels, y_pred, average='macro', zero_division=0)
        else:
            # Average per-class F1
            per_class_f1 = f1_score(labels, y_pred, average=None, zero_division=0)
            score = np.mean(per_class_f1)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


# Global threshold for compute_metrics (can be updated)
_current_threshold = 0.3

def compute_metrics(eval_pred):
    """
    Compute metrics for multi-label classification
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        Dictionary of metric scores
    """
    global _current_threshold
    predictions, labels = eval_pred
    
    # Apply sigmoid to logits to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions)).numpy()
    
    # Convert probabilities to binary predictions
    y_pred = (probs > _current_threshold).astype(int)
    y_true = labels
    
    # Calculate metrics
    metrics = {
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'accuracy_subset': accuracy_score(y_true, y_pred),  # Exact match accuracy
    }
    
    # Per-label F1 scores (for individual distortions)
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, dist_label in enumerate(COGNITIVE_DISTORTION_LABELS):
        if i < len(per_label_f1):
            metrics[f'f1_{dist_label}'] = float(per_label_f1[i])
    
    return metrics


def load_data(data_dir="data/processed/distortions"):
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
    
    # VERIFY LABEL INTEGRITY - Check if labels are all zeros
    print(f"\n🔍 Verifying label integrity...")
    print("-" * 70)
    
    # Check training labels
    train_labels = np.array([example['labels'] for example in train_dataset])
    val_labels = np.array([example['labels'] for example in val_dataset])
    
    print(f"Training set label statistics:")
    print(f"  Total examples: {len(train_labels)}")
    print(f"  Examples with all zeros: {np.sum(np.sum(train_labels, axis=1) == 0)}")
    print(f"  Examples with at least one positive: {np.sum(np.sum(train_labels, axis=1) > 0)}")
    print(f"  Total positive labels: {np.sum(train_labels)}")
    print(f"  Label shape: {train_labels.shape}")
    
    # Print sample labels
    print(f"\n  Sample labels (first 5 examples):")
    for i in range(min(5, len(train_labels))):
        print(f"    Example {i}: {train_labels[i].tolist()}")
        print(f"      Sum: {np.sum(train_labels[i])}, Non-zero indices: {np.where(train_labels[i] > 0)[0].tolist()}")
    
    # Check per-class positive counts
    print(f"\n  Per-class positive counts in training set:")
    pos_counts = np.sum(train_labels, axis=0)
    neg_counts = len(train_labels) - pos_counts
    for i, dist_label in enumerate(COGNITIVE_DISTORTION_LABELS):
        print(f"    {DISTORTION_NAMES[dist_label]:30s}: {int(pos_counts[i]):4d} positive, {int(neg_counts[i]):4d} negative")
    
    # Check if all labels are zeros
    if np.sum(train_labels) == 0:
        print("\n❌ ERROR: All training labels are zeros! Check data preprocessing.")
        raise ValueError("All training labels are zeros - cannot train model")
    
    if np.sum(val_labels) == 0:
        print("\n⚠️  WARNING: All validation labels are zeros!")
    
    print("-" * 70)
    
    return train_dataset, val_dataset, test_dataset, pos_counts, neg_counts


def load_model(model_name="roberta-base", num_labels=None):
    """
    Load pre-trained model for multi-label classification
    
    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of distortion labels (default: from cognitive_distortions)
    
    Returns:
        model, tokenizer
    """
    if num_labels is None:
        num_labels = get_num_distortions()
    
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
    output_dir="models/distortion_classifier",
    num_epochs=10,
    batch_size=16,
    learning_rate=1e-5,
    pos_weight=None
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
        pos_weight: Positive class weights for BCEWithLogitsLoss (tensor)
    
    Returns:
        Trainer object
    """
    print(f"\n⚙️  Setting up training...")
    
    # FIX CLASS IMBALANCE - Create custom loss function with pos_weight
    if pos_weight is not None:
        print(f"  Using pos_weight for class imbalance:")
        for i, dist_label in enumerate(COGNITIVE_DISTORTION_LABELS):
            print(f"    {DISTORTION_NAMES[dist_label]:30s}: {pos_weight[i]:.3f}")
        
        # Create custom loss function that handles device placement
        class WeightedBCELoss:
            def __init__(self, pos_weight):
                # Store pos_weight on CPU initially, will move to device when needed
                self.pos_weight = pos_weight
                self.loss_fn = None  # Will be created on first call with device info
            
            def __call__(self, logits, labels):
                # Get device from logits
                device = logits.device
                # Move pos_weight to same device as logits
                pos_weight_on_device = self.pos_weight.to(device)
                # Create loss function with pos_weight on correct device
                loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight_on_device)
                return loss_fn(logits, labels.float())
        
        loss_fn = WeightedBCELoss(pos_weight)
    else:
        loss_fn = None  # Use default
    
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
        warmup_steps=100,                # Gradual warmup (fewer steps for smaller dataset)
        lr_scheduler_type="linear",      # Linear decay
        
        # Evaluation
        eval_strategy="steps",           # Evaluate every N steps
        eval_steps=200,                  # Evaluate every 200 steps
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,              # Keep only 2 best checkpoints
        load_best_model_at_end=True,     # Load best model after training
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        report_to="none",               
        
        # Performance
        fp16=torch.cuda.is_available(),  
        dataloader_num_workers=4,
        
        # Reproducibility
        seed=42,
    )
    
    # Create Trainer with custom loss
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    
    # Override compute_loss if pos_weight is provided
    if pos_weight is not None:
        original_compute_loss = trainer.compute_loss
        
        def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Custom loss function with pos_weight support
            Args match Trainer's expected signature
            """
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
        
        trainer.compute_loss = compute_loss
    
    print(f"✓ Trainer configured:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Loss function: {'Weighted BCE' if pos_weight is not None else 'BCE'}")
    
    return trainer


def predict_sample(model, tokenizer, text, threshold=0.3, print_probs=False):
    """
    Predict cognitive distortions for a single text
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Input text
        threshold: Prediction threshold
        print_probs: Whether to print raw probabilities
    
    Returns:
        List of predicted distortion labels with probabilities, and raw probabilities
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
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # DIAGNOSE THRESHOLD - Print raw sigmoid probabilities
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    
    if print_probs:
        print(f"\n  Raw sigmoid probabilities (before thresholding):")
        for i, dist_label in enumerate(COGNITIVE_DISTORTION_LABELS):
            print(f"    {DISTORTION_NAMES[dist_label]:30s}: {probs[i]:.4f}")
        print(f"  Max probability: {np.max(probs):.4f}")
        print(f"  Mean probability: {np.mean(probs):.4f}")
    
    predictions = (probs > threshold).astype(int)
    
    # Get predicted distortion labels
    predicted_distortions = [
        (COGNITIVE_DISTORTION_LABELS[i], probs[i], DISTORTION_NAMES[COGNITIVE_DISTORTION_LABELS[i]]) 
        for i, pred in enumerate(predictions) if pred == 1
    ]
    
    # Sort by probability (descending)
    predicted_distortions.sort(key=lambda x: x[1], reverse=True)
    
    return predicted_distortions, probs


def test_model_predictions(model, tokenizer, threshold=0.3):
    """
    Test model on sample texts to verify it's working
    Includes raw probability diagnostics
    """
    print("\n🧪 Testing model on sample texts...")
    print("=" * 70)
    
    test_texts = [
        "I'm definitely going to fail this test tomorrow. Everything is ruined.",
        "I always mess everything up. I'm such a failure.",
        "They probably think I'm stupid. Everyone at the meeting must have thought I was incompetent.",
        "I should be able to handle this. I shouldn't make any mistakes.",
        "I feel anxious, so something bad must be about to happen.",
        "Today was actually pretty good. I got some work done and felt productive.",
    ]
    
    for idx, text in enumerate(test_texts):
        print(f"\n{'='*70}")
        print(f"Sample {idx + 1}: {text}")
        print("-" * 70)
        distortions, probs = predict_sample(model, tokenizer, text, threshold=threshold, print_probs=True)
        print(f"\nPredicted distortions (threshold={threshold}):")
        if distortions:
            for dist_label, prob, dist_name in distortions:
                print(f"  ✓ {dist_name} ({dist_label}): {prob:.3f}")
        else:
            print("  - No distortions detected (all probabilities below threshold)")
    
    print("=" * 70)


def train_model(
    data_dir="data/processed/distortions",
    output_dir="models/distortion_classifier",
    num_epochs=10,
    batch_size=16,
    learning_rate=1e-5
):
    """
    Main training pipeline
    
    Args:
        data_dir: Directory with preprocessed data
        output_dir: Directory to save trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    
    print("\n" + "="*70)
    print("🚀 COGNITIVE DISTORTION CLASSIFICATION TRAINING")
    print("="*70)
    
    # Step 1: Load data and verify labels
    train_dataset, val_dataset, test_dataset, pos_counts, neg_counts = load_data(data_dir)
    
    # Step 2: FIX CLASS IMBALANCE - Compute pos_weight
    print(f"\n⚖️  Computing class weights for imbalanced data...")
    pos_weight = []
    for i in range(len(COGNITIVE_DISTORTION_LABELS)):
        pos_count = pos_counts[i]
        neg_count = neg_counts[i]
        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0  # No positive examples for this class
        pos_weight.append(weight)
    
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
    print(f"✓ Computed pos_weight tensor: {pos_weight_tensor.tolist()}")
    
    # Step 3: Load model
    num_labels = get_num_distortions()
    model, tokenizer = load_model(num_labels=num_labels)
    
    # Step 4: Create trainer with pos_weight
    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pos_weight=pos_weight_tensor
    )
    
    # Step 5: Train
    print("\n🏋️  Starting training...")
    print("-" * 70)
    train_result = trainer.train()
    
    # Step 6: Save model
    print("\n💾 Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to {output_dir}")
    
    # Step 7: TUNE PREDICTION THRESHOLD - Find optimal threshold
    print("\n🎯 Finding optimal prediction threshold...")
    print("-" * 70)
    
    # Get predictions on validation set
    val_predictions = trainer.predict(val_dataset)
    val_probs = torch.sigmoid(torch.Tensor(val_predictions.predictions)).numpy()
    val_labels = np.array([example['labels'] for example in val_dataset])
    
    # Find optimal threshold
    optimal_threshold, optimal_score = find_optimal_threshold(val_probs, val_labels, metric='f1_micro')
    print(f"  Optimal threshold: {optimal_threshold:.3f} (F1-micro: {optimal_score:.4f})")
    
    # Evaluate with optimal threshold
    print(f"\n📊 Evaluating on validation set (threshold={optimal_threshold:.3f})...")
    
    # Update global threshold for compute_metrics
    global _current_threshold
    _current_threshold = optimal_threshold
    
    # Manually compute with optimal threshold
    val_pred_optimal = (val_probs > optimal_threshold).astype(int)
    f1_micro_opt = f1_score(val_labels, val_pred_optimal, average='micro', zero_division=0)
    f1_macro_opt = f1_score(val_labels, val_pred_optimal, average='macro', zero_division=0)
    per_label_f1_opt = f1_score(val_labels, val_pred_optimal, average=None, zero_division=0)
    
    print("\nValidation Results (with optimal threshold):")
    print("-" * 70)
    print("Overall Metrics:")
    print(f"  F1-micro: {f1_micro_opt:.4f}")
    print(f"  F1-macro: {f1_macro_opt:.4f}")
    print(f"  Precision-micro: {precision_score(val_labels, val_pred_optimal, average='micro', zero_division=0):.4f}")
    print(f"  Recall-micro: {recall_score(val_labels, val_pred_optimal, average='micro', zero_division=0):.4f}")
    print(f"  Subset accuracy: {accuracy_score(val_labels, val_pred_optimal):.4f}")
    
    print("\nPer-Distortion F1 Scores:")
    for i, dist_label in enumerate(COGNITIVE_DISTORTION_LABELS):
        if i < len(per_label_f1_opt):
            print(f"  {DISTORTION_NAMES[dist_label]:30s}: {per_label_f1_opt[i]:.4f}")
    
    # Step 8: Test predictions with optimal threshold
    print(f"\n🧪 Testing model predictions (threshold={optimal_threshold:.3f})...")
    test_model_predictions(model, tokenizer, threshold=optimal_threshold)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {output_dir}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Best validation F1-micro: {f1_micro_opt:.4f}")
    print(f"Best validation F1-macro: {f1_macro_opt:.4f}")
    
    return trainer, model, tokenizer, optimal_threshold


if __name__ == "__main__":
    """
    Run training
    Usage: python src/train_distortion.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train cognitive distortion classifier")
    parser.add_argument("--data_dir", type=str, default="data/processed/distortions",
                        help="Directory with preprocessed data")
    parser.add_argument("--output_dir", type=str, default="models/distortion_classifier",
                        help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    # Check if preprocessed data exists
    if not os.path.exists(args.data_dir):
        print("❌ Preprocessed data not found!")
        print(f"Please run preprocessing first:")
        print(f"  python src/preprocess_distortions.py")
        exit(1)
    
    # Train model
    trainer, model, tokenizer, optimal_threshold = train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
