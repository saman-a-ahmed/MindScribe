"""
Preprocessing pipeline for Go Emotions dataset
Handles: text cleaning, tokenization, multi-hot encoding, train/val split
"""

import re
from datasets import load_dataset, load_from_disk
from transformers import RobertaTokenizer
import os


class EmotionPreprocessor:
    """
    Preprocessor for emotion classification dataset
    - Cleans text (URLs, lowercase, whitespace)
    - Tokenizes using RoBERTa
    - Converts labels to multi-hot vectors
    """
    
    def __init__(self, model_name='roberta-base', max_length=128, num_labels=28):
        """
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length for tokenization
            num_labels: Number of emotion labels (28 for go_emotions)
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.num_labels = num_labels
        print(f"✓ Loaded {model_name} tokenizer")
        print(f"  Max length: {max_length}, Num labels: {num_labels}")
    
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
        text = text.strip()
        
        return text
    
    def labels_to_multihot(self, labels):
        """
        Convert label list to multi-hot binary vector
        Example: [2, 14] → [0.0, 0.0, 1.0, 0.0, ..., 1.0, 0.0, ...]
        
        Args:
            labels: List of integer label indices
        Returns:
            Multi-hot binary list of length num_labels (as floats)
        """
        multihot = [0.0] * self.num_labels  # Use 0.0 instead of 0
        for label in labels:
            if 0 <= label < self.num_labels:  # Validate label range
                multihot[label] = 1.0  # Use 1.0 instead of 1
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


def prepare_dataset(
    dataset_name="go_emotions",
    dataset_config="simplified",
    save_to_disk=True,
    output_dir="data/processed",
    model_name='roberta-base',
    max_length=128
):
    """
    Complete preprocessing pipeline:
    1. Load dataset
    2. Apply preprocessing (clean, tokenize, encode labels)
    3. Create train/val split
    4. Save to disk (optional)
    
    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        save_to_disk: Whether to save processed data locally
        output_dir: Directory to save processed data
        model_name: Tokenizer model name
        max_length: Maximum sequence length
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    
    # Step 1: Load dataset
    print(f"\n📂 Loading {dataset_name} ({dataset_config})...")
    dataset = load_dataset(dataset_name, dataset_config)
    
    print(f"✓ Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")
    
    # Step 2: Initialize preprocessor
    print(f"\n🔧 Initializing preprocessor...")
    preprocessor = EmotionPreprocessor(
        model_name=model_name,
        max_length=max_length,
        num_labels=28  # go_emotions has 28 emotion labels
    )
    
    # Step 3: Apply preprocessing with .map()
    print(f"\n⚙️  Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocessor.preprocess_function,
        batched=True,                    # Process in batches (faster)
        remove_columns=['text', 'id'],   # Remove original columns
        desc="Preprocessing"             # Progress bar description
    )
    
    # Step 3.5: Cast labels column to float explicitly
    from datasets import Features, Sequence, Value
    new_features = processed_dataset['train'].features.copy()
    new_features['labels'] = Sequence(Value('float32'))
    
    processed_dataset = processed_dataset.cast(new_features)
    print(f"✓ Labels cast to float32")
    
    # Step 4: Extract splits
    train_dataset = processed_dataset['train']
    val_dataset = processed_dataset['validation']
    test_dataset = processed_dataset['test']
    
    print(f"✓ Preprocessing complete")
    
    # Step 5: Save to disk (optional)
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
    print(f"\n📊 Sample Processed Example(s):")
    print("=" * 70)
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # Decode tokens back to text
        decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        
        print(f"\nSample {i + 1}:")
        print(f"  Decoded text: {decoded_text[:100]}...")  # First 100 chars
        print(f"  Input IDs (first 10): {sample['input_ids'][:10]}")
        print(f"  Attention mask (first 10): {sample['attention_mask'][:10]}")
        print(f"  Labels (multi-hot): {sample['labels']}")
        print(f"  Active emotions: {[i for i, val in enumerate(sample['labels']) if val == 1]}")
        print(f"  Sequence length: {len(sample['input_ids'])}")
    
    print("=" * 70)


def load_preprocessed_data(data_dir="data/processed"):
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
    print(f"  Val: {len(val)} examples")
    print(f"  Test: {len(test)} examples")
    
    return train, val, test


if __name__ == "__main__":
    """
    Run preprocessing pipeline. Run from the project root as a module:
      python -m src.preprocessing
    """

    # Run preprocessing
    train, val, test = prepare_dataset(
        save_to_disk=True,
        output_dir="data/processed"
    )
    
    # Print sample for verification
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    print_sample(train, tokenizer, num_samples=2)
    
    print("\n✅ Preprocessing complete!")
    print("\nTo load data later:")
    print("  from src.preprocessing import load_preprocessed_data")
    print("  train, val, test = load_preprocessed_data('data/processed')")