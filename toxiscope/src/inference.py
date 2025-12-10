#!/usr/bin/env python3
"""
ToxiScope: Standalone Inference Module

Simple inference interface for trained ToxiScope models.

Usage:
    python inference.py --text "You're such a noob!"
    python inference.py --model outputs/models/deberta_best --text "Test"
    python inference.py --file comments.txt --output predictions.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Default labels
DEFAULT_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "racism"]


class ToxiScopePredictor:
    """
    High-level inference interface for ToxiScope models.
    
    Works with any HuggingFace-compatible model saved with save_pretrained().
    """
    
    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        max_length: int = 256,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            model_dir: Path to saved model directory
            device: Device to use ('auto', 'cuda', 'cpu')
            max_length: Maximum sequence length
            labels: Label names (if not in config)
        """
        self.max_length = max_length
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Check for best_model subdirectory
        if (model_path / "best_model").exists():
            model_path = model_path / "best_model"
        
        print(f"Loading model from {model_path}...")
        
        # Load model and tokenizer using HuggingFace standard API
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Handle tokenizer loading with fallback for tiktoken compatibility
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except (ValueError, AttributeError):
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
        
        # Get labels
        if labels:
            self.labels = labels
        elif hasattr(self.model.config, 'label_names'):
            self.labels = self.model.config.label_names
        elif hasattr(self.model.config, 'id2label'):
            self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        else:
            # Try to load from labels.txt
            labels_path = model_path / "labels.txt"
            if labels_path.exists():
                self.labels = labels_path.read_text().strip().split('\n')
            else:
                self.labels = DEFAULT_LABELS
        
        # Load thresholds
        self.thresholds = self._load_thresholds(model_path)
        
        print(f"Loaded model with {len(self.labels)} labels: {self.labels}")
        print(f"Using device: {self.device}")
    
    def _load_thresholds(self, model_path: Path) -> Dict[str, float]:
        """Load thresholds from file or use defaults."""
        # Try multiple locations
        for path in [
            model_path / "thresholds.json",
            model_path.parent / "thresholds.json",
        ]:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        
        # Default thresholds
        return {label: 0.5 for label in self.labels}
    
    def predict(
        self,
        texts: List[str],
        return_probs: bool = True,
    ) -> List[Dict]:
        """
        Predict toxicity for a list of texts.
        
        Args:
            texts: List of input texts
            return_probs: Whether to include probabilities in output
        
        Returns:
            List of prediction dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Apply thresholds
        results = []
        for i, (text, prob_row) in enumerate(zip(texts, probs)):
            active_labels = []
            label_probs = {}
            
            for j, label in enumerate(self.labels):
                threshold = self.thresholds.get(label, 0.5)
                prob = float(prob_row[j])
                label_probs[label] = round(prob, 4)
                
                if prob >= threshold:
                    active_labels.append(label)
            
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "is_toxic": len(active_labels) > 0,
                "active_labels": active_labels,
            }
            
            if return_probs:
                result["probabilities"] = label_probs
            
            results.append(result)
        
        return results
    
    def predict_single(self, text: str, return_probs: bool = True) -> Dict:
        """Predict for a single text."""
        results = self.predict([text], return_probs=return_probs)
        return results[0]
    
    def batch_predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_probs: bool = True,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Predict for large batches efficiently.
        
        Args:
            texts: List of texts
            batch_size: Batch size for inference
            return_probs: Include probabilities
            show_progress: Show progress bar
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(texts), batch_size), desc="Predicting")
            except ImportError:
                iterator = range(0, len(texts), batch_size)
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch, return_probs=return_probs)
            results.extend(batch_results)
        
        return results


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="ToxiScope Inference")
    parser.add_argument("--model", type=str, default="outputs/models/deberta_base/best",
                        help="Path to model directory")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--file", type=str, help="File with texts (one per line)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Load predictor
    try:
        predictor = ToxiScopePredictor(args.model, device=args.device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train a model first or specify a valid model path.")
        sys.exit(1)
    
    # Process inputs
    if args.text:
        result = predictor.predict_single(args.text)
        print("\nPrediction:")
        print(json.dumps(result, indent=2))
    
    elif args.file:
        texts = Path(args.file).read_text().strip().split('\n')
        print(f"Processing {len(texts)} texts...")
        results = predictor.batch_predict(texts, batch_size=args.batch_size)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved to {args.output}")
        else:
            for r in results[:5]:
                print(json.dumps(r, indent=2))
            if len(results) > 5:
                print(f"... and {len(results) - 5} more")
    
    elif args.interactive:
        print("\n" + "="*60)
        print("ToxiScope Interactive Mode")
        print("Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                text = input("Enter text: ").strip()
                if text.lower() in ('quit', 'exit', 'q'):
                    break
                if not text:
                    continue
                
                result = predictor.predict_single(text)
                print(f"\n  Is toxic: {result['is_toxic']}")
                print(f"  Labels: {result['active_labels']}")
                print(f"  Probabilities:")
                for label, prob in result['probabilities'].items():
                    bar = "█" * int(prob * 20)
                    print(f"    {label:15s}: {prob:.3f} {bar}")
                print()
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
