# forge-workshop

AI Ignite at York — Forge Track, Workshop 01: **Emotion Detection** (BiLSTM vs BERT vs LLMs on MELD)

Friday 27 November 2026, 5:30 to 7:30 PM, WSC 101.

## Start here

Open the base notebook in Colab and run it top to bottom:

[forge_workshop_base.ipynb](https://colab.research.google.com/github/sagarh12/forge-workshop/blob/main/forge_workshop_base.ipynb)

## Repo layout

```
data/raw/               original MELD files (do not modify)
data/processed/         meld_clean.csv, meld_context.csv (Mehwish)
notebooks/davit/        zero-shot, few-shot, data balancing
notebooks/ali/          fine-tuning: BiLSTM and BERT
notebooks/sheheryar/    evaluation pipeline and scoreboard
outputs/predictions/    one CSV per model
outputs/reports/        classification reports
outputs/figures/        charts and confusion matrices
slides/                 decks
```

## Rules for every contributor

1. **Use MELD's official train / dev / test split.** Never call `train_test_split`. The base notebook creates `X_train`, `X_test`, `y_train`, `y_test` for you.
2. **Save predictions with `save_predictions()`** from the base notebook. Columns: `utterance`, `true_label`, `predicted_label`, `model_name`.
3. **Model names are fixed:** `bilstm`, `bert_finetuned`, `zero_shot`, `few_shot`, `balanced_few_shot`.
4. **Labels are lowercase:** `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`.
5. **Nothing heavy runs live.** Training and full test-set inference are pre-computed and committed.
6. **Your own Gemini key, never committed.** Each attendee creates a free key in Google AI Studio. Paste it at the prompt; never put it in a cell or a file.
7. **LLM calls are batched** (20 utterances per request) and live runs are capped at 100 rows.

## Data files (committed by Mehwish)

`data/processed/meld_clean.csv` — columns: `utterance`, `label`, `dialogue_id`, `utterance_id`, `speaker`, `split`.
`data/processed/meld_context.csv` — same, plus `context` (the previous three utterances of the dialogue).

## Contact

Sagar — sagarh@my.yorku.ca
