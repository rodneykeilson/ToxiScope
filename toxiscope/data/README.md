# ToxiScope Dataset Documentation

## Overview

The ToxiScope dataset consists of **893,525 Reddit comments** collected from 17 Indonesian gaming communities. This dataset supports multilabel toxicity classification with 7 distinct labels.

## Data Source

Comments were scraped from Reddit using [ScrapiReddit](https://github.com/rodneykeilson/ScrapiReddit), covering major Indonesian gaming communities including:

- DOTA2
- League of Legends
- Mobile Legends
- CS2 / CSGO
- Valorant
- PUBG Mobile
- Free Fire
- And more...

## Label Definitions

| Label | Description | Example |
|-------|-------------|---------|
| `toxic` | General toxic or negative content | "Dasar noob, uninstall aja!" |
| `severe_toxic` | Highly offensive, harmful content | [Extreme profanity/threats] |
| `obscene` | Vulgar or profane language | Contains explicit curse words |
| `threat` | Threats of violence or harm | "Awas lu ntar gw gebukin" |
| `insult` | Personal attacks or insults | "Lu emang goblok dari lahir" |
| `identity_hate` | Hate targeting identity groups | Discrimination based on identity |
| `racism` | Racially discriminatory content | Racial slurs or stereotypes |

## Statistics

### Dataset Size

```
Total comments:     893,525
Training set:       625,467 (70%)
Validation set:     134,029 (15%)
Test set:           134,029 (15%)
```

### Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| toxic | 87,240 | 9.76% |
| obscene | 54,036 | 6.05% |
| insult | 20,355 | 2.28% |
| racism | 1,057 | 0.12% |
| threat | 935 | 0.10% |
| identity_hate | 644 | 0.07% |
| severe_toxic | 534 | 0.06% |

Note the severe class imbalance - this motivates our use of focal loss.

### Text Length Distribution

```
Mean length (characters): 113.47
Median length (characters): 64.0
Mean length (tokens): 20.61
Median length (tokens): 12.0
```

## Data Splits

Data is split using **thread-stratified sampling** to prevent data leakage:
- Comments from the same Reddit post appear only in one split
- Multilabel stratification ensures label distribution consistency

## File Format

CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `comment_id` | Unique Reddit comment ID |
| `post_id` | Parent post ID |
| `subreddit` | Source subreddit |
| `body` | Original comment text |
| `body_clean` | Normalized text |
| `created_utc` | Timestamp |
| `score` | Reddit score |
| `toxic` | Binary (0/1) |
| `severe_toxic` | Binary (0/1) |
| `obscene` | Binary (0/1) |
| `threat` | Binary (0/1) |
| `insult` | Binary (0/1) |
| `identity_hate` | Binary (0/1) |
| `racism` | Binary (0/1) |

## Usage

### Loading the Data

```python
import pandas as pd

# Load preprocessed splits
train_df = pd.read_csv("data/processed/train.csv", dtype=str)
val_df = pd.read_csv("data/processed/val.csv", dtype=str)
test_df = pd.read_csv("data/processed/test.csv", dtype=str)

# Label columns
LABELS = ["toxic", "severe_toxic", "obscene", "threat", 
          "insult", "identity_hate", "racism"]

# Extract features and labels
X_train = train_df["body_clean"].tolist()
y_train = train_df[LABELS].values.astype(int)
```

### Preprocessing Pipeline

The `body_clean` column is preprocessed with:
1. URL removal
2. Mention removal (@username)
3. Emoji removal
4. Whitespace normalization
5. Deleted comment filtering

For IndoBERT, additional preprocessing is minimal as the model handles raw text.

## Labeling Methodology

Initial labels were assigned using rule-based pattern matching with curated regex patterns in `patterns/` directory. Each pattern file contains:

- Indonesian toxic words and phrases
- Common slang variants
- Regional expressions (Javanese, Sundanese, etc.)

Human validation was performed on a sample to calibrate thresholds.

## Ethical Considerations

- This dataset contains offensive content for research purposes only
- Models trained on this data should be deployed responsibly
- Consider cultural context when interpreting Indonesian toxicity
- Regular updates needed to capture evolving slang

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{toxiscope2024,
  title={ToxiScope Indonesian Toxicity Dataset},
  author={Your Name},
  year={2024},
  note={893,525 Reddit comments with multilabel toxicity annotations}
}
```

## License

This dataset is released for research purposes under [LICENSE].

## Contact

For questions or issues, please contact [email@example.com].
