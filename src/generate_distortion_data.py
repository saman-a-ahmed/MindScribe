"""
Synthetic Dataset Generator for Cognitive Distortions
Generates training examples using pattern-based generation + optional LLM enhancement
"""

import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from cognitive_distortions import (
    COGNITIVE_DISTORTION_LABELS,
    EXAMPLE_TEXTS,
    NEUTRAL_EXAMPLES,
    DISTORTION_DESCRIPTIONS,
    SIGNAL_WORDS
)


class DistortionDataGenerator:
    """
    Generate synthetic training data for cognitive distortion detection
    Uses pattern-based generation with variations
    """
    
    def __init__(self, seed=42):
        """Initialize generator with random seed"""
        random.seed(seed)
        self.distortion_labels = COGNITIVE_DISTORTION_LABELS
        self.num_distortions = len(self.distortion_labels)
    
    def generate_variations(self, base_text: str, distortion_type: str, num_variations: int = 3) -> List[str]:
        """
        Generate variations of a base text with the same distortion
        
        Args:
            base_text: Original example text
            distortion_type: Type of cognitive distortion
            num_variations: Number of variations to generate
            
        Returns:
            List of variation texts
        """
        variations = [base_text]  # Include original
        
        # Generate simple variations by substituting signal words
        signal_words = SIGNAL_WORDS.get(distortion_type, [])
        
        for _ in range(num_variations - 1):
            variation = base_text.lower()
            
            # Try to replace similar signal words
            if signal_words:
                # Simple word replacement variations
                replacements = {
                    'always': ['constantly', 'every time', 'continuously'],
                    'never': ['not ever', 'at no time'],
                    'everything': ['every single thing', 'all things'],
                    'nothing': ['not a thing', 'no thing'],
                    'everyone': ['everybody', 'all people'],
                    'should': ['must', 'ought to', 'have to'],
                    'terrible': ['awful', 'horrible', 'dreadful'],
                    'disaster': ['catastrophe', 'ruined', 'destroyed'],
                }
                
                for word, alternatives in replacements.items():
                    if word in variation and alternatives:
                        variation = variation.replace(word, random.choice(alternatives), 1)
                        break
            
            variations.append(variation)
        
        return variations[:num_variations]
    
    def generate_multi_label_example(self, distortion_types: List[str]) -> str:
        """
        Generate text with multiple distortions
        
        Args:
            distortion_types: List of distortion labels to combine
            
        Returns:
            Combined text example
        """
        if not distortion_types:
            return random.choice(NEUTRAL_EXAMPLES)
        
        # Get example from each distortion type
        examples = []
        for dist_type in distortion_types:
            if dist_type in EXAMPLE_TEXTS:
                examples.append(random.choice(EXAMPLE_TEXTS[dist_type]))
        
        if len(examples) == 1:
            return examples[0]
        
        # Combine examples naturally
        # Simple approach: join with "Also," or "And"
        if len(examples) == 2:
            connectors = [" Also, ", " And ", " Plus, ", " "]
            return examples[0] + random.choice(connectors) + examples[1].lower()
        else:
            combined = examples[0]
            for ex in examples[1:]:
                combined += random.choice([" Also, ", " And ", " "]) + ex.lower()
            return combined
    
    def generate_single_label_examples(
        self,
        distortion_type: str,
        num_examples: int = 20,
        variations_per_example: int = 3
    ) -> List[Tuple[str, List[int]]]:
        """
        Generate examples for a single distortion type
        
        Args:
            distortion_type: Distortion label
            num_examples: Number of base examples to generate
            variations_per_example: Variations per base example
            
        Returns:
            List of (text, label_vector) tuples
        """
        examples = []
        base_texts = EXAMPLE_TEXTS.get(distortion_type, [])
        
        # Get distortion index
        dist_idx = self.distortion_labels.index(distortion_type)
        
        # Generate from base examples
        for base_text in base_texts:
            variations = self.generate_variations(base_text, distortion_type, variations_per_example)
            for var_text in variations:
                # Create multi-hot label vector (only this distortion active)
                label_vector = [0] * self.num_distortions
                label_vector[dist_idx] = 1
                examples.append((var_text, label_vector))
        
        # Fill remaining slots with pattern-based generation
        while len(examples) < num_examples:
            if base_texts:
                base_text = random.choice(base_texts)
                variation = self.generate_variations(base_text, distortion_type, 1)[0]
                label_vector = [0] * self.num_distortions
                label_vector[dist_idx] = 1
                examples.append((variation, label_vector))
            else:
                break
        
        return examples[:num_examples]
    
    def generate_multi_label_examples(
        self,
        num_examples: int = 50,
        max_distortions_per_example: int = 3
    ) -> List[Tuple[str, List[int]]]:
        """
        Generate examples with multiple distortions
        
        Args:
            num_examples: Number of examples to generate
            max_distortions_per_example: Maximum distortions per example
            
        Returns:
            List of (text, label_vector) tuples
        """
        examples = []
        
        for _ in range(num_examples):
            # Choose number of distortions (1 to max_distortions_per_example)
            num_dist = random.randint(1, max_distortions_per_example)
            
            # Choose distortion types (without replacement)
            distortion_types = random.sample(self.distortion_labels, min(num_dist, len(self.distortion_labels)))
            
            # Generate text
            text = self.generate_multi_label_example(distortion_types)
            
            # Create multi-hot label vector
            label_vector = [0] * self.num_distortions
            for dist_type in distortion_types:
                dist_idx = self.distortion_labels.index(dist_type)
                label_vector[dist_idx] = 1
            
            examples.append((text, label_vector))
        
        return examples
    
    def generate_neutral_examples(self, num_examples: int = 30) -> List[Tuple[str, List[int]]]:
        """
        Generate neutral examples (no distortions)
        
        Args:
            num_examples: Number of neutral examples
            
        Returns:
            List of (text, label_vector) tuples
        """
        examples = []
        label_vector = [0] * self.num_distortions  # All zeros
        
        # Use base neutral examples and create variations
        for _ in range(num_examples):
            text = random.choice(NEUTRAL_EXAMPLES)
            # Simple variations: change tense, add minor words
            variations = [
                text,
                text.replace('today', random.choice(['yesterday', 'recently', 'this week'])),
                text.replace('I', random.choice(['I', 'I\'ve', 'I have'])),
            ]
            examples.append((random.choice(variations), label_vector))
        
        return examples
    
    def generate_full_dataset(
        self,
        examples_per_distortion: int = 20,
        multi_label_examples: int = 50,
        neutral_examples: int = 30,
        variations_per_example: int = 3
    ) -> pd.DataFrame:
        """
        Generate complete synthetic dataset
        
        Args:
            examples_per_distortion: Examples per single distortion type
            multi_label_examples: Examples with multiple distortions
            neutral_examples: Examples with no distortions
            variations_per_example: Variations per base example
            
        Returns:
            DataFrame with 'text' and 'labels' columns
        """
        print("🔄 Generating synthetic cognitive distortion dataset...")
        print("=" * 70)
        
        all_examples = []
        
        # Generate single-label examples (one distortion per text)
        print(f"\n📝 Generating single-label examples...")
        for i, distortion_type in enumerate(self.distortion_labels, 1):
            print(f"  [{i}/{len(self.distortion_labels)}] {distortion_type}...", end=" ")
            examples = self.generate_single_label_examples(
                distortion_type,
                num_examples=examples_per_distortion,
                variations_per_example=variations_per_example
            )
            all_examples.extend(examples)
            print(f"✓ {len(examples)} examples")
        
        # Generate multi-label examples
        print(f"\n📝 Generating multi-label examples...")
        print(f"  Generating {multi_label_examples} examples with multiple distortions...", end=" ")
        multi_examples = self.generate_multi_label_examples(
            num_examples=multi_label_examples,
            max_distortions_per_example=3
        )
        all_examples.extend(multi_examples)
        print(f"✓ {len(multi_examples)} examples")
        
        # Generate neutral examples
        print(f"\n📝 Generating neutral examples...")
        print(f"  Generating {neutral_examples} neutral examples...", end=" ")
        neutral_exs = self.generate_neutral_examples(num_examples=neutral_examples)
        all_examples.extend(neutral_exs)
        print(f"✓ {len(neutral_exs)} examples")
        
        # Shuffle
        random.shuffle(all_examples)
        
        # Create DataFrame
        texts = [ex[0] for ex in all_examples]
        labels = [ex[1] for ex in all_examples]
        
        df = pd.DataFrame({
            'text': texts,
            'labels': labels
        })
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   Total examples: {len(df)}")
        print(f"   Examples per distortion:")
        for i, dist_label in enumerate(self.distortion_labels):
            count = sum(1 for label_vec in labels if label_vec[i] == 1)
            print(f"     - {dist_label}: {count}")
        
        neutral_count = sum(1 for label_vec in labels if sum(label_vec) == 0)
        print(f"     - neutral (no distortion): {neutral_count}")
        
        print("=" * 70)
        
        return df


def save_dataset(df: pd.DataFrame, output_path: str, train_split: float = 0.7, val_split: float = 0.15):
    """
    Save dataset to disk with train/val/test split
    
    Args:
        df: DataFrame with 'text' and 'labels' columns
        output_path: Directory to save datasets
        train_split: Proportion for training (default 0.7)
        val_split: Proportion for validation (default 0.15)
        # test_split: 1 - train_split - val_split
    """
    from pathlib import Path
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    n = len(df)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Save as JSON (easy to load)
    train_path = output_path / "train.json"
    val_path = output_path / "val.json"
    test_path = output_path / "test.json"
    
    train_df.to_json(train_path, orient='records', indent=2)
    val_df.to_json(val_path, orient='records', indent=2)
    test_df.to_json(test_path, orient='records', indent=2)
    
    print(f"\n💾 Saved datasets to {output_path}:")
    print(f"   Train: {len(train_df)} examples → {train_path}")
    print(f"   Val:   {len(val_df)} examples → {val_path}")
    print(f"   Test:  {len(test_df)} examples → {test_path}")


if __name__ == "__main__":
    """
    Generate synthetic dataset for cognitive distortion detection
    Usage: python src/generate_distortion_data.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic cognitive distortion dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/distortions",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--examples_per_distortion",
        type=int,
        default=20,
        help="Number of examples per single distortion type"
    )
    parser.add_argument(
        "--multi_label_examples",
        type=int,
        default=50,
        help="Number of examples with multiple distortions"
    )
    parser.add_argument(
        "--neutral_examples",
        type=int,
        default=30,
        help="Number of neutral examples"
    )
    parser.add_argument(
        "--variations_per_example",
        type=int,
        default=3,
        help="Number of variations per base example"
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = DistortionDataGenerator(seed=42)
    df = generator.generate_full_dataset(
        examples_per_distortion=args.examples_per_distortion,
        multi_label_examples=args.multi_label_examples,
        neutral_examples=args.neutral_examples,
        variations_per_example=args.variations_per_example
    )
    
    # Save
    save_dataset(
        df,
        output_path=args.output_dir,
        train_split=0.7,
        val_split=0.15
    )
    
    print("\n✅ Dataset generation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review generated data: {args.output_dir}/")
    print(f"  2. Run preprocessing: python src/preprocess_distortions.py")
    print(f"  3. Train model: python src/train_distortion.py")
