# ToxiScope 🔬

**Deep Learning-Based Toxic Comment Detection for Gaming Communities**

> Formerly known as *Commulyzer* - Now powered by BERT-tiny transformer for efficient CPU inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

ToxiScope is a multilabel toxic comment detection system optimized for **gaming community discourse**. It uses BERT-tiny (4.4M parameters) fine-tuned with Focal Loss to handle severe class imbalance, achieving strong performance on toxicity detection while remaining efficient enough for CPU inference.

### Model Performance

| Metric | Score |
|--------|-------|
| **Macro-F1** | **0.783** |
| **Micro-F1** | **0.873** |
| Training Time | ~8 minutes (CPU) |
| Parameters | 4.4M |

### Per-Label Results (with Calibrated Thresholds)

| Label | F1 Score | Threshold | ROC-AUC |
|-------|----------|-----------|---------|
| toxic | 0.895 | 0.60 | 0.974 |
| severe_toxic | 0.700 | 0.50 | 0.954 |
| obscene | 0.945 | 0.50 | 0.988 |
| threat | 0.457 | 0.40 | 0.930 |
| insult | 0.873 | 0.45 | 0.978 |
| identity_hate | 0.818 | 0.60 | 0.948 |
| racism | 0.794 | 0.60 | 0.977 |

**Training Details:**
- Model: `prajjwal1/bert-tiny` (HuggingFace)
- Loss: Focal Loss (γ=2.0, α=0.25)
- Dataset: 10K training samples, 2K validation samples
- Epochs: ~3 (early stopping based on best validation loss)

### Key Features

- 🎮 **Gaming Community Focus**: Trained on Reddit gaming subreddits (Dota 2, CS2, LoL, Valorant, etc.)
- 📊 **Multilabel Classification**: Detects 7 toxicity categories simultaneously
- ⚡ **Fast CPU Inference**: BERT-tiny enables <100ms predictions on CPU
- 🎯 **Focal Loss**: Handles severe class imbalance effectively
- 🔬 **Calibrated Thresholds**: Per-label optimized thresholds for best F1
- 📈 **10K Training Samples**: Efficient training in minutes, not hours

### Toxicity Labels

| Label | Description | % of Dataset |
|-------|-------------|--------------|
| `toxic` | General toxic/negative content | 9.76% |
| `severe_toxic` | Highly offensive or harmful content | 0.06% |
| `obscene` | Vulgar or profane language | 5.32% |
| `threat` | Threats of violence or harm | 0.10% |
| `insult` | Personal attacks or insults | 4.83% |
| `identity_hate` | Hate speech targeting identity groups | 0.61% |
| `racism` | Racially discriminatory content | 0.52% |

## 🏗️ Project Structure

```
toxiscope/
├── configs/                    # Training & model configurations
│   ├── deberta_base.yaml      # DeBERTa-v3-base config (recommended)
│   ├── deberta_large.yaml     # DeBERTa-v3-large config
│   ├── roberta_toxicity.yaml  # RoBERTa baseline
│   ├── bilstm_cnn.yaml        # BiLSTM-CNN ablation
│   └── sweep_config.yaml      # WandB hyperparameter sweep
├── data/                       # Preprocessed dataset splits
│   └── README.md              # Data documentation
├── src/
│   ├── models/                # Model architectures
│   │   ├── deberta.py        # DeBERTa-v3 classifier with enhancements
│   │   ├── bilstm_cnn.py     # BiLSTM-CNN baseline
│   │   └── focal_loss.py     # Class-imbalance handling
│   ├── preprocessing/         # Text processing pipeline
│   │   ├── normalizer.py     # English text normalization
│   │   └── augmentation.py   # Data augmentation for minorities
│   ├── train.py              # Training script with WandB
│   ├── evaluate.py           # Comprehensive evaluation
│   ├── inference.py          # Production inference
│   └── export_onnx.py        # ONNX export for browser deployment
├── notebooks/
│   └── 03_ablation_study.ipynb  # Model comparison & figures
├── journal/                   # LaTeX paper skeleton
│   └── main.tex
├── outputs/
│   ├── models/               # Saved checkpoints
│   └── reports/              # Metrics & figures
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone and navigate
cd toxiscope

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Training BERT-tiny

```bash
# Train with BERT-tiny (recommended for CPU)
python src/train.py --config configs/bert_tiny.yaml

# Train with DistilBERT (requires more time/memory)
python src/train.py --config configs/distilbert_base.yaml

# With WandB tracking
python src/train.py --config configs/bert_tiny.yaml --wandb
```

### Evaluation

```bash
# Evaluate trained model
python src/evaluate.py \
    --model outputs/models/bert_tiny/best_model \
    --test data/processed/test.csv
```

### Quick Inference

```bash
# Command line inference
python src/inference.py \
    --model outputs/models/bert_tiny/best_model \
    --text "You're such a noob, uninstall the game"
```

```python
# Python API
from src.inference import ToxiScopePredictor

# Load model
predictor = ToxiScopePredictor("outputs/models/bert_tiny/best_model")

# Predict
result = predictor.predict("You're such a noob, uninstall the game")
print(result)
# {
#   "is_toxic": True,
#   "active_labels": ["toxic", "insult"],
#   "probabilities": {"toxic": 0.78, "insult": 0.71, ...}
# }
```

## 📊 Model Comparison

| Model | Macro-F1 | Micro-F1 | Parameters | Training Time |
|-------|----------|----------|------------|---------------|
| TF-IDF + LogReg (baseline) | 0.858 | 0.957 | ~50K | seconds |
| **BERT-tiny + Focal Loss** | **0.783** | **0.873** | **4.4M** | **~8 min (CPU)** |

*Note: The baseline model uses full dataset (600K+ samples) while BERT-tiny uses 10K samples for demonstration. Training on larger datasets will improve performance.*

### Per-Label Performance Comparison

| Label | Baseline F1 | BERT-tiny F1 |
|-------|-------------|--------------|
| toxic | 0.957 | 0.997 | 0.65 |
| severe_toxic | 0.795 | 1.000 | 0.50 |
| obscene | 0.972 | 0.998 | 0.65 |
| threat | 0.706 | 0.999 | 0.95 |
| insult | 0.945 | 0.998 | 0.75 |
| identity_hate | 0.730 | 0.980 | 0.65 |
| racism | 0.903 | 0.988 | 0.45 |

## 🔬 Ablation Studies

Compare different configurations:

### Pre-trained Models
- DeBERTa-v3-base (recommended)
- DeBERTa-v3-large
- RoBERTa-base
- BERT-base (baseline)
- BiLSTM-CNN (non-transformer baseline)

### Pooling Strategies
- CLS token pooling
- Mean pooling
- Attention pooling (learned)

### Loss Functions
- Binary Cross-Entropy
- Focal Loss (γ=2.0)
- Asymmetric Loss
- Weighted BCE

## 🌐 Chrome Extension

Real-time toxicity detection for Reddit and other platforms:

```bash
# Export optimized model
python src/export_onnx.py --checkpoint best --quantize

# Load extension in Chrome
# 1. Open chrome://extensions
# 2. Enable Developer mode
# 3. Load unpacked → select extension/ folder
```

**Performance:**
- Inference: <200ms per comment
- Model size: ~50MB (quantized)
- Runtime: ONNX Runtime Web

## 📚 Citation

If you use ToxiScope in your research, please cite:

```bibtex
@article{toxiscope2024,
  title={ToxiScope: DeBERTa-v3 Based Multilabel Toxic Comment Detection for Gaming Communities},
  author={Author Name},
  journal={Journal Name},
  year={2024}
}
```

## 📖 References

- He et al. (2021) "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" - ACL 2021
- He et al. (2023) "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training" - ICLR 2023
- Lin et al. (2017) "Focal Loss for Dense Object Detection" - ICCV 2017
- Jigsaw Toxic Comment Classification Challenge (2018) - Kaggle

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
