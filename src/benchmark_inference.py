"""
Inference Benchmark Script for ToxiScope
Measures average inference time per text for different models
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Sample texts for benchmarking
BENCHMARK_TEXTS = [
    "You're such a noob, uninstall the game",
    "Great match everyone, GG!",
    "I'm going to kill you in the next round",
    "This team is absolutely garbage, trash players everywhere",
    "Nice shot! You're really good at this game",
    "Go back to playing Minecraft you stupid kid",
    "I hate playing with randoms, they always feed",
    "Thanks for the carry, you're amazing!",
    "Report this player for hacking, so obvious",
    "Let's group up and push together",
    "You're the worst player I've ever seen",
    "Good game, close one!",
    "Stop feeding or I'll report you",
    "Can we surrender? This is hopeless",
    "That was an insane play, well done!",
]

def load_model(model_path: str, device: str = 'cpu'):
    """Load model and tokenizer"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load thresholds
    threshold_path = Path(model_path) / "thresholds.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds = json.load(f)
    else:
        thresholds = {label: 0.5 for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'racism']}
    
    return model, tokenizer, thresholds

def benchmark_single_inference(model, tokenizer, thresholds, texts, device='cpu', n_runs=100):
    """Benchmark single-text inference (extension/real-time scenario)"""
    
    times = []
    
    for _ in tqdm(range(n_runs), desc="Single inference benchmark"):
        text = np.random.choice(texts)
        
        start = time.perf_counter()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'median_ms': np.median(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
    }

def benchmark_batch_inference(model, tokenizer, thresholds, texts, device='cpu', batch_sizes=[1, 8, 16, 32]):
    """Benchmark batch inference for different batch sizes"""
    
    results = {}
    
    for batch_size in batch_sizes:
        times = []
        n_runs = max(10, 100 // batch_size)
        
        for _ in tqdm(range(n_runs), desc=f"Batch={batch_size}"):
            batch = [np.random.choice(texts) for _ in range(batch_size)]
            
            start = time.perf_counter()
            
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
            
            end = time.perf_counter()
            times.append((end - start) * 1000 / batch_size)  # ms per text
        
        results[batch_size] = {
            'mean_ms_per_text': np.mean(times),
            'std_ms_per_text': np.std(times),
            'throughput_texts_per_sec': 1000 / np.mean(times)
        }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--n-runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default='outputs/reports/inference_benchmark.json')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model, tokenizer, thresholds = load_model(args.model, args.device)
    
    # Get model info
    model_params = sum(p.numel() for p in model.parameters())
    model_name = args.model.split('/')[-1]
    
    print(f"\nModel: {model_name}")
    print(f"Parameters: {model_params:,}")
    print(f"Device: {args.device}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        inputs = tokenizer("warmup text", return_tensors="pt", truncation=True, max_length=128).to(args.device)
        with torch.no_grad():
            model(**inputs)
    
    # Single inference benchmark
    print("\nRunning single-text inference benchmark...")
    single_results = benchmark_single_inference(
        model, tokenizer, thresholds, BENCHMARK_TEXTS,
        device=args.device, n_runs=args.n_runs
    )
    
    # Batch inference benchmark
    print("\nRunning batch inference benchmark...")
    batch_results = benchmark_batch_inference(
        model, tokenizer, thresholds, BENCHMARK_TEXTS,
        device=args.device
    )
    
    # Compile results
    results = {
        'model': model_name,
        'model_path': args.model,
        'parameters': model_params,
        'device': args.device,
        'single_inference': single_results,
        'batch_inference': batch_results
    }
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE BENCHMARK RESULTS")
    print("="*60)
    print(f"\nModel: {model_name}")
    print(f"Parameters: {model_params:,}")
    print(f"\nSingle Text Inference (real-time scenario):")
    print(f"  Mean: {single_results['mean_ms']:.2f} ms")
    print(f"  Median: {single_results['median_ms']:.2f} ms")
    print(f"  Std: {single_results['std_ms']:.2f} ms")
    print(f"  P95: {single_results['p95_ms']:.2f} ms")
    print(f"  P99: {single_results['p99_ms']:.2f} ms")
    print(f"\nBatch Inference:")
    for batch_size, metrics in batch_results.items():
        print(f"  Batch {batch_size}: {metrics['mean_ms_per_text']:.2f} ms/text ({metrics['throughput_texts_per_sec']:.1f} texts/sec)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
