"""
Preprocessing pipeline for Cognitive Distortions dataset
Handles: text cleaning, tokenization, multi-hot encoding, train/val split
"""

import re
import json
import pandas as pd
from pathlib import Path
from datasets import Dataset, load_from_disk
from transformers import RobertaTokenizer
import os
from src.cognitive_distortions import COGNITIVE_DISTORTION_LABELS, get_num_distortions


class DistortionPreprocessor:
    """
    Preprocessor for cognitive distortion classification dataset
    - Cleans text (URLs, lowercase, whitespace)
    - Tokenizes using RoBERTa
    - Converts labels to multi-hot vectors
    """
    
    def __init__(self, model_name='roberta-base', max_length=128, num_labels=None):
        """
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length for tokenization
            num_labels: Number of distortion labels (default: from cognitive_distortions)
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.num_labels = num_labels if num_labels else get_num_distortions()
        print(f"✓ Loaded {model_name} tokenizer")
        print(f"  Max length: {max_length}, Num labels: {self.num_labels}")
    
    def clean_text(self, text):
        """
        Clean raw text:
        1. Remove URLs (http/www links)
        2. Convert to lowercase
        3. Strip extra whitespace
        
        Args:
            text: Raw text string
        Returns:
            Cleaned text string
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Lowercase for standardization
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        text = text.strip()
        
        return text
    
    def labels_to_multihot(self, labels):
        """
        Convert label list to multi-hot binary vector
        Example: [0, 1, 0, 1, ...] → [0.0, 1.0, 0.0, 1.0, ...]
        
        Args:
            labels: List of binary labels (0/1) or list of distortion indices
        Returns:
            Multi-hot binary list of length num_labels (as floats)
        """
        # If labels is already a list of 0/1 (multi-hot), just convert to float
        if isinstance(labels, list) and len(labels) == self.num_labels:
            # Check if it's already binary
            if all(isinstance(x, (int, float)) and x in [0, 1] for x in labels):
                return [float(x) for x in labels]
        
        # If labels is a list of indices, convert to multi-hot
        multihot = [0.0] * self.num_labels
        if isinstance(labels, list):
            for label in labels:
                if isinstance(label, int) and 0 <= label < self.num_labels:
                    multihot[label] = 1.0
        elif isinstance(labels, int) and 0 <= labels < self.num_labels:
            multihot[labels] = 1.0
        
        return multihot
    
    def preprocess_function(self, examples):
        """
        Main preprocessing function for dataset.map()
        Processes batches of examples
        
        Args:
            examples: Dictionary with 'text' and 'labels' keys (batched)
        Returns:
            Dictionary with tokenized inputs and multi-hot labels
        """
        # Step 1: Clean all texts in batch
        cleaned_texts = [self.clean_text(text) for text in examples['text']]
        
        # Step 2: Tokenize batch
        tokenized = self.tokenizer(
            cleaned_texts,
            padding='max_length',      # Pad to max_length
            truncation=True,            # Truncate if longer
            max_length=self.max_length,
            return_tensors=None         # Return lists (not tensors)
        )
        
        # Step 3: Convert labels to multi-hot encoding
        tokenized['labels'] = [
            self.labels_to_multihot(labels) 
            for labels in examples['labels']
        ]
        
        return tokenized


def load_json_dataset(json_path: str) -> pd.DataFrame:
    """
    Load dataset from JSON file
    
    Args:
        json_path: Path to JSON file with 'text' and 'labels' keys
        
    Returns:
        DataFrame with 'text' and 'labels' columns
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    return df


def prepare_dataset(
    data_dir="data/raw/distortions",
    save_to_disk=True,
    output_dir="data/processed/distortions",
    model_name='roberta-base',
    max_length=128
):
    """
    Complete preprocessing pipeline:
    1. Load dataset from JSON files
    2. Apply preprocessing (clean, tokenize, encode labels)
    3. Create train/val/test splits
    4. Save to disk (optional)
    
    Args:
        data_dir: Directory containing train.json, val.json, test.json
        save_to_disk: Whether to save processed data locally
        output_dir: Directory to save processed data
        model_name: Tokenizer model name
        max_length: Maximum sequence length
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from datasets import Features, Sequence, Value
    
    # Step 1: Load JSON files
    data_dir = Path(data_dir)
    print(f"\n📂 Loading datasets from {data_dir}...")
    
    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"
    test_path = data_dir / "test.json"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    train_df = load_json_dataset(train_path)
    val_df = load_json_dataset(val_path)
    test_df = load_json_dataset(test_path)
    
    print(f"✓ Datasets loaded:")
    print(f"  Train: {len(train_df)} examples")
    print(f"  Val:   {len(val_df)} examples")
    print(f"  Test:  {len(test_df)} examples")
    
    # Step 2: Convert to HuggingFace Dataset format
    print(f"\n🔄 Converting to HuggingFace Dataset format...")
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Step 3: Initialize preprocessor
    print(f"\n🔧 Initializing preprocessor...")
    num_labels = get_num_distortions()
    preprocessor = DistortionPreprocessor(
        model_name=model_name,
        max_length=max_length,
        num_labels=num_labels
    )
    
    # Step 4: Apply preprocessing with .map()
    print(f"\n⚙️  Preprocessing datasets...")
    
    def preprocess_wrapper(examples):
        return preprocessor.preprocess_function(examples)
    
    train_dataset = train_dataset.map(
        preprocess_wrapper,
        batched=True,
        remove_columns=['text'],  # Keep labels for now, will be replaced
        desc="Preprocessing train"
    )
    
    val_dataset = val_dataset.map(
        preprocess_wrapper,
        batched=True,
        remove_columns=['text'],
        desc="Preprocessing val"
    )
    
    test_dataset = test_dataset.map(
        preprocess_wrapper,
        batched=True,
        remove_columns=['text'],
        desc="Preprocessing test"
    )
    
    # Step 5: Cast labels to float32 explicitly
    print(f"✓ Casting labels to float32...")
    label_features = Sequence(Value('float32'), length=num_labels)
    
    train_features = train_dataset.features.copy()
    train_features['labels'] = label_features
    train_dataset = train_dataset.cast(train_features)
    
    val_features = val_dataset.features.copy()
    val_features['labels'] = label_features
    val_dataset = val_dataset.cast(val_features)
    
    test_features = test_dataset.features.copy()
    test_features['labels'] = label_features
    test_dataset = test_dataset.cast(test_features)
    
    print(f"✓ Preprocessing complete")
    
    # Step 6: Save to disk (optional)
    if save_to_disk:
        print(f"\n💾 Saving to disk...")
        os.makedirs(output_dir, exist_ok=True)
        
        train_dataset.save_to_disk(f"{output_dir}/train")
        val_dataset.save_to_disk(f"{output_dir}/val")
        test_dataset.save_to_disk(f"{output_dir}/test")
        
        print(f"✓ Saved to {output_dir}/")
        print(f"  - train: {output_dir}/train")
        print(f"  - val:   {output_dir}/val")
        print(f"  - test:  {output_dir}/test")
    
    return train_dataset, val_dataset, test_dataset


def print_sample(dataset, tokenizer, num_samples=1):
    """
    Print sample(s) from processed dataset for verification
    
    Args:
        dataset: Processed dataset
        tokenizer: Tokenizer to decode tokens
        num_samples: Number of samples to print
    """
    from src.cognitive_distortions import COGNITIVE_DISTORTION_LABELS

    print(f"\n📊 Sample Processed Example(s):")
    print("=" * 70)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # Decode tokens back to text
        decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        
        # Get active distortions
        active_distortions = [
            COGNITIVE_DISTORTION_LABELS[j]
            for j, val in enumerate(sample['labels'])
            if val == 1.0
        ]
        
        print(f"\nSample {i + 1}:")
        print(f"  Decoded text: {decoded_text[:100]}...")
        print(f"  Input IDs (first 10): {sample['input_ids'][:10]}")
        print(f"  Attention mask (first 10): {sample['attention_mask'][:10]}")
        print(f"  Labels (multi-hot): {sample['labels']}")
        print(f"  Active distortions: {active_distortions if active_distortions else ['none (neutral)']}")
        print(f"  Sequence length: {len(sample['input_ids'])}")
    
    print("=" * 70)


def load_preprocessed_data(data_dir="data/processed/distortions"):
    """
    Load previously preprocessed data from disk
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    print(f"📂 Loading preprocessed data from {data_dir}...")
    
    train = load_from_disk(f"{data_dir}/train")
    val = load_from_disk(f"{data_dir}/val")
    test = load_from_disk(f"{data_dir}/test")
    
    print(f"✓ Loaded:")
    print(f"  Train: {len(train)} examples")
    print(f"  Val:   {len(val)} examples")
    print(f"  Test:  {len(test)} examples")
    
    return train, val, test


if __name__ == "__main__":
    """
    Run preprocessing pipeline. Run from the project root as a module:
      python -m src.preprocess_distortions
    """
    
    # Check if raw data exists
    raw_data_dir = "data/raw/distortions"
    if not Path(raw_data_dir).exists():
        print("❌ Raw data not found!")
        print(f"Please generate the dataset first:")
        print(f"  python -m src.generate_distortion_data")
        exit(1)
    
    # Run preprocessing
    train, val, test = prepare_dataset(
        data_dir=raw_data_dir,
        save_to_disk=True,
        output_dir="data/processed/distortions"
    )
    
    # Print sample for verification
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    print_sample(train, tokenizer, num_samples=2)
    
    print("\n✅ Preprocessing complete!")
    print("\nTo load data later:")
    print("  from src.preprocess_distortions import load_preprocessed_data")
    print("  train, val, test = load_preprocessed_data('data/processed/distortions')")
