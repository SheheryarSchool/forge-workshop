# MELD Dataset Preparation

## Dataset Source
MELD (Multimodal EmotionLines Dataset)
Official website: https://affective-meld.github.io/

## Dataset Overview
Total utterances: 13,708

Official splits:
- Train: 9,989
- Dev: 1,109
- Test: 2,610

## Emotion Labels
anger, disgust, fear, joy, neutral, sadness, surprise

## Cleaning Process
1. Loaded the official MELD train, dev, and test CSV files.
2. Preserved the original dataset split assignments.
3. Selected and renamed the required columns.
4. Standardized emotion labels to lowercase.
5. Corrected common text encoding artifacts.
6. Normalized whitespace in utterances and speaker names.
7. Verified that there were no missing values or
   duplicate dialogue positions.

## Output Files

### meld_clean.csv
Columns:
utterance, label, dialogue_id, utterance_id, speaker, split

### meld_context.csv
Contains all columns from meld_clean.csv, plus context.

Context includes up to three preceding utterances
from the same dialogue, formatted with speaker names.

Dialogue boundaries are defined using both split
and dialogue_id to prevent cross-split contamination.

The first utterance of each dialogue has empty context.

## Important
Use MELD's official train/dev/test splits.

Do not use random train_test_split on this dataset.
