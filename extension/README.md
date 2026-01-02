# ToxiScope Browser Extension 🔬

**Real-time Gaming Toxicity Detection for Chrome/Firefox**

This extension scans the visible text on any page, runs ToxiScope's toxicity classifier locally, and replaces or highlights toxic sentences in real-time. All inference happens in the browser—no network calls or server components required.

## ✨ Features

- **7 Toxicity Labels**: Detects toxic, severe_toxic, obscene, threat, insult, identity_hate, and racism
- **Popup Dashboard**: View statistics, detection counts, and toxicity rates
- **Customizable Settings**: 
  - Toggle detection on/off
  - Switch between placeholder and highlight modes
  - Rescan page on demand
- **Real-time Processing**: MutationObserver for dynamically loaded content
- **Zero Latency**: Pure JavaScript inference, no API calls

## 📊 Model Architecture

### Why TF-IDF for Browser Extension?

The extension uses a **TF-IDF + Logistic Regression** baseline model for practical browser deployment:

| Aspect | TF-IDF Baseline | BERT-tiny Transformer |
|--------|----------------|----------------------|
| **Model Size** | ~2MB (JSON) | ~20MB (ONNX) |
| **Inference Speed** | <1ms/sentence | ~50-100ms/sentence |
| **Memory Usage** | ~10MB | ~100-200MB |
| **Dependencies** | None (pure JS) | ONNX Runtime (~5MB) |
| **Initialization** | <100ms | 2-5 seconds |

For browser extensions that must load instantly and process hundreds of comments without lag, the TF-IDF baseline is the practical choice.

### Performance

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.858 |
| Micro-F1 | 0.957 |
| Inference Time | <1ms per sentence |
| Model Load Time | <100ms |

## 🚀 Project Structure

```
extension/
├── manifest.json          # Chrome/Firefox Manifest V3
├── src/
│   ├── content.ts         # Main content script (DOM processing)
│   ├── popup.html         # Extension popup UI
│   ├── popup.ts           # Popup logic and statistics
│   └── model/
│       ├── artifacts.ts   # Model loading from chrome.runtime.getURL
│       ├── inference.ts   # TF-IDF vectorization + logistic regression
│       └── types.ts       # TypeScript interfaces
├── assets/model/          # TF-IDF model artifacts (JSON)
│   ├── vocabulary_combined.json
│   ├── classifier_coefficients.json
│   ├── classifier_intercepts.json
│   ├── thresholds.json
│   └── metadata.json
└── dist/                  # Compiled JavaScript
```

## 📦 Getting Started

```powershell
cd extension
npm install
npm run build
```

To export the baseline model from Python:

```powershell
python -m scripts.baseline.export_to_json --artifacts-dir outputs/models/baseline --output-dir extension/assets/model
```

After building, load the `extension` folder as an unpacked extension:
- **Chrome/Edge**: `chrome://extensions` → Enable Developer Mode → Load Unpacked
- **Firefox**: `about:debugging` → This Firefox → Load Temporary Add-on

## ⚙️ How It Works

1. **Artifact Loading**: Lazily loads TF-IDF vocabulary and classifier weights from `chrome.runtime.getURL(...)`
2. **DOM Walking**: Uses `TreeWalker` to collect text nodes, splitting them into sentences
3. **TF-IDF Vectorization**: Converts sentences to sparse TF-IDF vectors using the vocabulary
4. **Logistic Regression**: Computes sigmoid(weights · features + intercept) for each label
5. **Threshold Application**: Applies per-label calibrated thresholds
6. **DOM Modification**: Replaces/highlights toxic sentences with label information
7. **Mutation Observer**: Reprocesses dynamically inserted content with debouncing
8. **Statistics Tracking**: Counts scanned sentences, toxic detections, and label distribution

## 🎨 Display Modes

### Placeholder Mode (Default)
```html
<span class="toxiscope-mask">
  <this sentence was removed for toxic, obscene>
</span>
```

### Highlight Mode
```html
<span class="toxiscope-highlight" data-toxicity-labels="toxic, insult" title="Detected: toxic, insult">
  Original toxic sentence here
</span>
```

## 📜 Development Scripts

- `npm run build` – Compile TypeScript to `dist/`
- `npm run watch` – Rebuild on changes for rapid iteration
- `npm run lint` – Run ESLint on source files

## 📄 License

MIT

## Notes

- The bundled artifacts can be large (vocabulary JSON is ~20 MB). Chrome can load large static assets but initial processing may take a few seconds on low-end hardware.
- Replace the placeholder icon under `icons/icon128.png` with your preferred artwork before distributing the extension.
