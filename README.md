<p align="center">
  <img src="assets/logo/ToxiScope.jpg" width="200" alt="ToxiScope Logo">
</p>

<h1 align="center">ToxiScope</h1>
<p align="center"><b>Efficient Transformer-Based Multilabel Toxic Comment Detection for Gaming Communities</b></p>
<p align="center">[Rodney Keilson](https://github.com/rodneykeilson), Felix Willie, and Dylan Pratama Khu</p>

ToxiScope is a multilabel toxicity detection system for gaming communities, featuring a BERT-tiny transformer fine-tuned with Focal Loss for efficient CPU inference.

## Results

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

## Project Structure

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
└── requirements.txt
```

## Toxicity Labels

| Label | Description | Training % |
|-------|-------------|------------|
| `toxic` | General toxic/negative content | 17.92% |
| `severe_toxic` | Highly offensive content | 0.73% |
| `obscene` | Vulgar or profane language | 9.22% |
| `threat` | Threats of violence | 1.24% |
| `insult` | Personal attacks | 4.65% |
| `identity_hate` | Hate speech targeting identity | 0.84% |
| `racism` | Racially discriminatory content | 1.32% |

## Quick Start

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

## Documentation

- [Extension README](extension/README.md) - Browser extension guide
- [Mobile App README](app/README.md) - React Native app guide

## Architecture

### Why BERT-tiny?

We use BERT-tiny (`prajjwal1/bert-tiny`) with 4.4M parameters:

| Model | Parameters | CPU Training | Inference | F1 |
|-------|-----------|--------------|-----------|-----|
| **BERT-tiny** | 4.4M | ~26 min | <10ms | 0.817 |
| DistilBERT | 66M | ~8 hrs | ~30ms | ~0.83 |
| BERT-base | 110M | ~16 hrs | ~50ms | ~0.84 |
| DeBERTa-v3 | 184M | ~24 hrs | ~80ms | ~0.86 |

Advantages:
- **CPU Training**: No GPU required
- **Fast Iteration**: 5-10 experiments/day
- **Memory Efficient**: <2GB RAM
- **Real-time Ready**: Sub-10ms inference

### Focal Loss

We use Focal Loss to handle class imbalance:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

With γ=2.0 and α=0.25, this down-weights easy examples and focuses on hard negatives.

## References

1. D. Kwak and J. Blackburn, "Linguistic Analysis of Toxic Behavior in an Online Video Game," in *Social Informatics*, Springer, 2015.
2. T. Davidson, D. Warmsley, M. Macy, and I. Weber, "Automated hate speech detection and the problem of offensive language," in *Proc. ICWSM*, 2017.
3. P. He, X. Liu, J. Gao, and W. Chen, "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," in *Proc. ICLR*, 2021.
4. Jigsaw, "Toxic comment classification challenge," *Kaggle Competition*, 2018.
5. M. Mozafari, R. Farahbakhsh, and N. Crespi, "A BERT-based transfer learning approach for hate speech detection in online social media," in *Complex Networks and Their Applications VIII*, Springer, 2020.
6. T. Caselli, V. Basile, J. Mitrovic, and M. Granitzer, "HateBERT: Retraining BERT for abusive language detection in English," in *Proc. 5th Workshop on Online Abuse and Harms*, 2021.
7. L. Hanu and Unitary team, "Detoxify," *GitHub repository*, 2020.
8. Y. Liu, M. Ott, N. Goyal, et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," *arXiv preprint arXiv:1907.11692*, 2019.
9. K. Shores, Y. He, K. Swanenburg, et al., "The Identification of Deviance and its Impact on Retention in a Multiplayer Game," in *Proc. CSCW*, 2014.
10. R. Martens, J. Hamari, H. Breidung, and A. Becker, "Going green: How to study gamer behavior in-game," in *Proc. DiGRA*, 2015.
11. T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal loss for dense object detection," in *Proc. ICCV*, 2017.
12. T. Ridnik, E. Ben-Baruch, N. Zamir, et al., "Asymmetric loss for multi-label classification," in *Proc. ICCV*, 2021.
13. Z. C. Lipton, C. Elkan, and B. Narayanaswamy, "Optimal thresholding of classifiers to maximize F1 measure," in *ECML PKDD*, 2014.
14. I. Turc, M. Chang, K. Lee, and K. Toutanova, "Well-Read Students Learn Better: On the Importance of Pre-training Compact Models," *arXiv preprint arXiv:1908.08962*, 2019.
15. H. Inoue, "Multi-sample dropout for accelerated training and better generalization," *arXiv preprint arXiv:1905.09788*, 2019.

## Citation

```bibtex
@article{toxiscope2026,
  title={ToxiScope: Efficient Transformer-Based Multilabel Toxic Comment Detection for Gaming Communities},
  author={Rodney Keilson and Felix Willie and Dylan Pratama Khu},
  journal={Universitas Mikroskil},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
