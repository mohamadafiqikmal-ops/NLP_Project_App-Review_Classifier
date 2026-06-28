# NLP Project: App Review Classifier

Theme: 6; App Review Classifier

Title: Thread App Review Sentiment Analysis

A sentiment analysis pipeline for app reviews (App reviews translated from Indonesian), comparing classical ML models against fine-tuned BERT, with results explored through an interactive Streamlit dashboard.

## Team Members

| Name | Matrics Number |
|---|---|
| Muhammad Farkhin bin Abd Baser | A24AI0055 |
| Muhammad Haziem Azfar bin Mohd Ransaimon | A24AI0057 |
| Mohamad Afiq Ikmal Bin Mohd Roslan | A24AI0046 |
| Fadzlee Adam Bin Mohd Nazlee | A24AI0027 |

## Project Structure

```
.
├── notebooks/
│   ├── balanced_app_reviews.csv             # Labeled dataset used for training
│   ├── Rating_labeled.csv                   # Raw/intermediate labeled data
│   ├── sentiment_analysis_pipeline_training.ipynb  # Classical ML pipeline (BoW/TF-IDF + NB/LogReg)
│   ├── bert_sentiment_training.ipynb        # BERT fine-tuning pipeline
│   └── data_distribution.ipynb              # EDA / class distribution analysis
├── streamlit/
│   ├── main.py                              # Dashboard app
│   ├── models/                              # Exported vectorizers, encoders, and trained models
│   └── cm_bert.png                          # BERT confusion matrix
├── requirements.txt
└── README.md
```

## Approach

Two modeling tracks were trained on the same labeled review dataset (Positive / Negative / Neutral):

1. **Classical ML** — text is cleaned and stemmed (NLTK), then vectorized with Bag-of-Words and TF-IDF, and classified with Naive Bayes and Logistic Regression.
2. **BERT** — `bert-base-uncased` fine-tuned end-to-end on the same labels for comparison against the classical baselines.

Trained models and vectorizers are exported to `streamlit/models/` and loaded by the dashboard for live inference and evaluation visuals.

## Streamlit Dashboard

The app (`streamlit/main.py`) has six tabs:

- **Home** — project overview
- **Text Analyzer** — run live predictions on custom text input
- **Data Explorer** — browse and filter the underlying review dataset
- **Visualizations** — class distribution, word clouds, top features per class
- **Model Info** — metrics and confusion matrices for the classical models
- **BERT (Experimental)** — BERT-based predictions, if model weights are present

> **Note:** `streamlit/models/bert_sentiment_model/` currently only contains the tokenizer and config files, not the trained weights file (e.g. `model.safetensors`). Run the `bert_sentiment_training.ipynb` notebook to generate and save it before the BERT tab will work.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the dashboard:
   ```bash
   streamlit run streamlit/main.py
   ```

To retrain the models, run the notebooks under `notebooks/` in order — classical pipeline first, then BERT — and the exported artifacts will be saved into `streamlit/models/`.
