# ToxiScope 🔬

**Efficient Transformer-Based Multilabel Toxic Comment Detection for Gaming Communities**

ToxiScope is a multilabel toxicity detection system for gaming communities, featuring BERT-tiny transformer fine-tuned with Focal Loss for efficient CPU inference.

## 🏆 Results

| Metric | Score |
|--------|-------|
| **Macro-F1** | **0.817** |
| **Micro-F1** | **0.889** |
| Training Time | ~26 min (CPU) |
| Inference | <10ms |

### Per-Label Performance

| Label | F1 Score | ROC-AUC | Threshold |
|-------|----------|---------|-----------|
| toxic | 0.903 | 0.978 | 0.50 |
| severe_toxic | 0.744 | 0.932 | 0.40 |
| obscene | 0.942 | 0.991 | 0.30 |
| threat | 0.654 | 0.979 | 0.45 |
| insult | 0.881 | 0.993 | 0.50 |
| identity_hate | 0.766 | 0.976 | 0.55 |
| racism | 0.828 | 0.992 | 0.50 |

## 📁 Project Structure

```
ToxiScope/
├── src/                        # Deep learning training code
│   ├── train.py               # HuggingFace Trainer pipeline
│   ├── evaluate.py            # Model evaluation
│   ├── inference.py           # Run predictions
│   ├── visualize.py           # Confusion matrices, ROC curves
│   ├── models/                # Model architectures
│   │   └── focal_loss.py     # Focal Loss for class imbalance
│   └── preprocessing/         # Data preprocessing
├── configs/                    # Training configurations (YAML)
│   ├── bert_tiny.yaml         # BERT-tiny config
│   ├── bert_tiny_medium.yaml  # BERT-tiny on 50K dataset
│   └── ...
├── scripts/                    # Utility scripts
│   ├── data_pipeline/         # Data processing
│   │   ├── merge_comments.py
│   │   ├── label_comments.py
│   │   ├── clean_comments.py
│   │   └── create_sample_dataset.py
│   ├── baseline/              # TF-IDF baseline model
│   │   ├── inference.py
│   │   └── export_to_json.py
│   └── export/                # Exporting results
│       └── results_to_latex.py
├── notebooks/                  # Jupyter notebooks
│   └── 03_ablation_study.ipynb
├── extension/                  # Chrome browser extension (ToxiScope Mask)
│   ├── src/                   # TypeScript source (content.ts, popup.ts)
│   ├── assets/model/          # TF-IDF model (JSON)
│   └── manifest.json
├── app/                        # React Native mobile app
├── data/                       # Datasets
│   ├── raw/                   # Scraped Reddit data
│   ├── processed/             # Cleaned and labeled
│   └── training/              # Train/val/test splits
├── patterns/                   # Regex pattern libraries (TSV)
├── outputs/                    # Trained models & reports
├── journal/                    # LaTeX research paper
└── requirements.txt
```

## 🎮 Toxicity Labels

| Label | Description | Training % |
|-------|-------------|------------|
| `toxic` | General toxic/negative content | 17.92% |
| `severe_toxic` | Highly offensive content | 0.73% |
| `obscene` | Vulgar or profane language | 9.22% |
| `threat` | Threats of violence | 1.24% |
| `insult` | Personal attacks | 4.65% |
| `identity_hate` | Hate speech targeting identity | 0.84% |
| `racism` | Racially discriminatory content | 1.32% |

## 🚀 Quick Start

### Installation

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Training BERT-tiny

```powershell
# Train with default config
python src/train.py --config configs/bert_tiny.yaml

# Train on 50K dataset
python src/train.py --config configs/bert_tiny_medium.yaml
```

### Inference

```powershell
# Transformer inference
python src/inference.py --text "your comment here"

# Baseline TF-IDF inference
python -m scripts.baseline.inference --text "your comment here"
```

```python
# Python API
from src.inference import ToxiScopePredictor

predictor = ToxiScopePredictor("outputs/models/bert_tiny/best_model")
result = predictor.predict("You're such a noob, uninstall the game")
# {"is_toxic": True, "active_labels": ["toxic", "insult"], ...}
```

### Chrome Extension

```powershell
cd extension
npm install
npm run build
# Load unpacked extension in Chrome
```

### Data Pipeline

```powershell
# Merge scraped Reddit comments
python -m scripts.data_pipeline.merge_comments

# Label with toxicity patterns
python -m scripts.data_pipeline.label_comments --input data/processed/merged/merged_comments.csv

# Clean and deduplicate
python -m scripts.data_pipeline.clean_comments --input data/processed/merged/merged_comments_labeled.csv
```

## 📖 Documentation

- [Extension README](extension/README.md) - Browser extension guide
- [Mobile App README](app/README.md) - React Native app guide
- [Research Paper](journal/final_journal.tex) - Full methodology and results

## 🔬 Architecture

### Why BERT-tiny?

We use BERT-tiny (`prajjwal1/bert-tiny`) [1] with 4.4M parameters:

| Model | Parameters | CPU Training | Inference | F1 |
|-------|-----------|--------------|-----------|-----|
| **BERT-tiny** | 4.4M | ~26 min | <10ms | 0.817 |
| DistilBERT | 66M | ~8 hrs | ~30ms | ~0.83 |
| BERT-base | 110M | ~16 hrs | ~50ms | ~0.84 |
| DeBERTa-v3 | 184M | ~24 hrs | ~80ms | ~0.86 |

Advantages:
- ✅ **CPU Training**: No GPU required
- ✅ **Fast Iteration**: 5-10 experiments/day
- ✅ **Memory Efficient**: <2GB RAM
- ✅ **Real-time Ready**: Sub-10ms inference

### Focal Loss

We use Focal Loss [2] to handle class imbalance:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

With γ=2.0 and α=0.25, this down-weights easy examples and focuses on hard negatives.

## 📚 References

1. I. Turc et al., "Well-Read Students Learn Better: On the Importance of Pre-training Compact Models," arXiv:1908.08962, 2019.
2. T.Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
3. P. He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," ICLR 2021.
4. T. Davidson et al., "Automated Hate Speech Detection and the Problem of Offensive Language," ICWSM 2017.
5. D. Kwak and J. Blackburn, "Linguistic Analysis of Toxic Behavior in an Online Video Game," Social Informatics, 2015.

## 📝 Citation

```bibtex
@article{toxiscope2026,
  title={ToxiScope: Efficient Transformer-Based Multilabel Toxic Comment Detection for Gaming Communities},
  author={Keilson, Rodney and Willie, Felix and Khu, Dylan Pratama},
  journal={Universitas Mikroskil},
  year={2026}
}
```

## 📄 License

MIT
