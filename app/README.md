# ToxiScope Mobile App 📱

**On-Device Toxicity Detection for Gaming Communities**

This Expo + React Native app runs ToxiScope's toxicity detector entirely on-device, providing instant analysis without requiring network connectivity.

## ✨ Features

- **7 Toxicity Labels**: Detects toxic, severe_toxic, obscene, threat, insult, identity_hate, and racism
- **Visual Toxicity Meter**: Color-coded percentage indicator
- **Label Badges**: Clear display of detected toxicity categories with confidence scores  
- **Score Breakdown**: Sorted bar chart of all label probabilities
- **Offline-First**: All inference happens locally on the device
- **Cross-Platform**: Works on iOS, Android, and Web

## 📱 Screenshots

The app features a modern dark theme with:
- **Toxicity Meter**: Green (safe) → Yellow (mild) → Red (toxic)
- **Detected Categories**: Color-coded badges showing triggered labels
- **Score Breakdown**: Visual bars for each label's probability

## 🚀 Getting Started

```powershell
cd app
npm install
npm run start
```

Expo CLI opens the developer tools:
- Press `a` to launch Android emulator
- Press `i` for iOS simulator (macOS only)
- Press `w` to open the web preview

## 📦 Model Architecture

The app uses a **TF-IDF + Logistic Regression** baseline model for fast on-device inference:

| Component | Details |
|-----------|---------|
| **Vectorization** | TF-IDF with unigrams and bigrams |
| **Classifier** | One-vs-Rest Logistic Regression |
| **Labels** | 7 toxicity categories |
| **Model Size** | ~2MB (JSON artifacts) |
| **Inference Speed** | <10ms per text |

### Why Not Transformers on Mobile?

While BERT-tiny achieves higher accuracy, the TF-IDF baseline is preferred for mobile:
- **Cold Start**: <100ms vs 2-5 seconds for ONNX model loading
- **Memory**: ~10MB vs 100-200MB for transformer inference
- **Battery**: Minimal CPU usage vs sustained GPU/NPU activity
- **Bundle Size**: 2MB JSON vs 20MB+ ONNX weights

## 📁 Project Structure

```
app/
├── App.tsx                    # Main entry point
├── index.js                   # Expo entry
├── assets/
│   ├── icons/                 # App icons
│   └── model/                 # TF-IDF model artifacts
│       ├── metadata.json
│       ├── labels.txt
│       ├── thresholds.json
│       ├── classifier_coefficients.json
│       ├── classifier_intercepts.json
│       └── vocabulary_combined.json
└── src/
    ├── components/
    │   └── PrimaryButton.tsx   # Reusable button component
    ├── context/
    │   └── ModelContext.tsx    # React context for model state
    ├── model/
    │   ├── inference.ts        # TF-IDF vectorization + logistic regression
    │   ├── loadArtifacts.ts    # Asset loading and caching
    │   └── types.ts            # TypeScript interfaces
    └── screens/
        └── HomeScreen.tsx      # Main analysis interface
```

## 🔧 Model Assets

Copy the JSON exports from the baseline model training:

```powershell
# From the project root
python export_baseline_to_json.py --artifacts-dir outputs/models/baseline --output-dir app/assets/model
```

Required files:
- `metadata.json` - Model configuration (tokenization settings, etc.)
- `labels.txt` - List of toxicity labels
- `thresholds.json` - Per-label calibrated decision thresholds
- `classifier_coefficients.json` - Logistic regression weights
- `classifier_intercepts.json` - Logistic regression biases
- `vocabulary_combined.json` - TF-IDF vocabulary mapping

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Macro-F1** | 0.72 |
| **Micro-F1** | 0.85 |
| **Inference Time** | <10ms |
| **Memory Usage** | ~10MB |

## 🎨 UI Components

### HomeScreen
The main interface featuring:
- **Status Bar**: Model loading indicator with color-coded status
- **Text Input**: Multi-line input for gaming comments
- **Analyze Button**: Triggers on-device inference
- **Results Display**:
  - Toxicity meter with percentage
  - Active label badges with confidence
  - Score breakdown bars for all labels

### PrimaryButton
Reusable styled button with loading state support.

## 📜 Scripts

- `npm run start` - Start Expo development server
- `npm run android` - Run on Android emulator
- `npm run ios` - Run on iOS simulator
- `npm run web` - Run in web browser

## 📄 License

MIT
