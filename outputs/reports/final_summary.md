# ToxiScope Training Summary

**Date:** December 18, 2025  
**Model:** prajjwal1/bert-tiny (4.4M parameters)  
**Dataset:** 50K training / 10K validation samples  

## Final Results

### Overall Metrics
| Metric | Score |
|--------|-------|
| **Macro-F1** | **0.8167** |
| **Micro-F1** | **0.8894** |
| Training Time | ~26 minutes |
| Epochs Completed | 1.76 (early stopped at best checkpoint) |

### Per-Label Results (Calibrated Thresholds)

| Label | F1 Score | ROC-AUC | Threshold | Support (%) |
|-------|----------|---------|-----------|-------------|
| toxic | 0.903 | 0.978 | 0.50 | 17.92% |
| severe_toxic | 0.744 | 0.932 | 0.40 | 0.73% |
| obscene | 0.942 | 0.991 | 0.30 | 9.22% |
| threat | 0.654 | 0.979 | 0.45 | 1.24% |
| insult | 0.881 | 0.993 | 0.50 | 4.65% |
| identity_hate | 0.766 | 0.976 | 0.55 | 0.84% |
| racism | 0.828 | 0.992 | 0.50 | 1.32% |

## Training Configuration

- **Loss Function:** Focal Loss (γ=2.0, α=0.25)
- **Optimizer:** AdamW (lr=5×10⁻⁴)
- **Scheduler:** Linear warmup (10%) + decay
- **Batch Size:** 32
- **Max Length:** 128 tokens
- **Evaluation:** Every 500 steps
- **Best Model:** Saved by macro_f1 metric

## Achievements

1. ✅ **All 7 labels detected** - Non-zero F1 on all categories including rare classes
2. ✅ **Focal Loss effective** - Improved minority class detection (threat: 0.654, severe_toxic: 0.744)
3. ✅ **Fast training** - Only 26 minutes on CPU for 50K samples
4. ✅ **Strong ROC-AUC** - >0.93 on all labels indicating good ranking capability
5. ✅ **Calibrated thresholds** - Per-label optimization improves F1 vs fixed 0.5

## Generated Artifacts

### Model Files
- `outputs/models/deberta/best_model/` - Trained model checkpoint
  - `model.safetensors` - Model weights
  - `config.json` - Model configuration
  - `tokenizer.json` - Tokenizer
  - `thresholds.json` - Calibrated thresholds
  - `labels.txt` - Label ordering

### Reports & Visualizations
- `outputs/reports/error_analysis` - Markdown error analysis report
- `outputs/reports/error_analysis.json` - Raw error data
- `outputs/reports/figures/confusion_matrices.png` - Per-label confusion matrices
- `outputs/reports/figures/roc_curves.png` - ROC curves for all labels
- `outputs/reports/figures/threshold_sensitivity.png` - F1 vs threshold plots
- `outputs/reports/figures/optimal_thresholds.json` - Computed optimal thresholds

## Error Analysis Summary

### Most Common Errors

| Label | False Positives | False Negatives |
|-------|-----------------|-----------------|
| toxic | 205 | 188 |
| obscene | 82 | 46 |
| threat | 60 | 50 |
| insult | 58 | 75 |
| racism | 20 | 32 |
| identity_hate | 15 | 35 |
| severe_toxic | 7 | 33 |

### Patterns
- **False Negatives:** Sarcasm, gaming-specific slang, implicit threats
- **False Positives:** Quoted toxic content, competitive gaming terms, context-dependent phrases

## Comparison with Baseline

| Model | Macro-F1 | Micro-F1 | Inference |
|-------|----------|----------|-----------|
| TF-IDF + LR | 0.72 | 0.85 | <1ms |
| **BERT-tiny** | **0.817** | **0.889** | <10ms |
| Improvement | +13.5% | +4.6% | - |

## Next Steps (Optional)

1. Train DistilBERT for comparison (configs ready)
2. Export to ONNX for browser deployment
3. Add more gaming-specific training data
4. Implement sarcasm detection module
