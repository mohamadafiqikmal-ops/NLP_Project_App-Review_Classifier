import streamlit as st
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io, base64, re

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(
    page_title="Syntax : Indonesian App Review Analyser",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body { font-family: 'Inter', sans-serif; }
.stApp { background: #06060e; }
section[data-testid="stSidebar"] { background: #0b0b18; }
html, body, [class*="css"], p, div, span, label { color: #d4d4e8; }
h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e1e38;
    gap: 0;
    padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #555577;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 14px 22px;
    border: none;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #c4b5fd !important; }
.stTabs [data-baseweb="tab-panel"] { background: transparent; padding-top: 28px; }

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background: #0f0f22 !important;
    border: 1px solid #2a2a50 !important;
    color: #d4d4e8 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.12) !important;
}

.stButton button {
    background: #a78bfa;
    color: #06060e;
    border: none;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-radius: 8px;
    padding: 10px 28px;
    transition: background 0.18s;
}
.stButton button:hover { background: #c4b5fd; }
.stDataFrame { background: #0f0f22; border-radius: 10px; }
hr { border-color: #1e1e38 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #06060e; }
::-webkit-scrollbar-thumb { background: #2a2a50; border-radius: 3px; }

.metric-card {
    background: #0f0f22;
    border: 1px solid #1e1e38;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #555577;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 6px;
}

.section-label {
    font-size: 0.68rem;
    color: #a78bfa;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 6px;
}

.prediction-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.02em;
}

.model-card {
    background: #0f0f22;
    border: 1px solid #1e1e38;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 12px;
}
.model-card h4 {
    color: #a78bfa;
    margin: 0 0 6px 0;
    font-size: 1rem;
}
.model-card p {
    color: #9090b0;
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR   = BASE_DIR / "models"
DATA_PATH   = PROJECT_DIR / "notebooks" / "balanced_app_reviews.csv"
BERT_MODEL_PATH = PROJECT_DIR / "streamlit" / "models" / "bert_sentiment_model"

PLOT_THEME = {
    'plot_bgcolor':  'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#d4d4e8', 'family': 'Inter'},
    'xaxis': {'gridcolor': '#1e1e38', 'linecolor': '#2a2a50', 'tickfont': {'color': '#9090b0'}},
    'yaxis': {'gridcolor': '#1e1e38', 'linecolor': '#2a2a50', 'tickfont': {'color': '#9090b0'}},
    'legend': {'bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#d4d4e8'}},
}

SENTIMENT_COLORS = {
    "Positive": "#34d399",
    "Neutral":  "#a78bfa",
    "Negative": "#f87171",
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    review_col_candidates = ["translation", "clean_content", "content", "review_description", "text", "review"]
    sentiment_col_candidates = ["labelScore", "sentiment", "label", "rating_label"]
    rating_col_candidates = ["score", "rating", "stars"]
    date_col_candidates = ["at", "review_date", "date", "timestamp"]
    source_col_candidates = ["app", "source", "platform"]

    for candidates, target in [
        (review_col_candidates, "review_description"),
        (sentiment_col_candidates, "sentiment"),
        (rating_col_candidates, "rating"),
        (date_col_candidates, "review_date"),
        (source_col_candidates, "source"),
    ]:
        if target not in df.columns:
            for candidate in candidates:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: target})
                    break

    if "review_description" not in df.columns:
        text_cols = df.select_dtypes(include="object").columns.tolist()
        if text_cols:
            df = df.rename(columns={text_cols[0]: "review_description"})
        else:
            df["review_description"] = ""

    df["review_description"] = df["review_description"].fillna("").astype(str)

    if "sentiment" not in df.columns:
        df["sentiment"] = "Neutral"
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip().str.capitalize()
    df.loc[df["sentiment"] == "", "sentiment"] = "Neutral"

    df["text_length"] = df["review_description"].str.split().str.len().fillna(0).astype(int)
    return df

df = load_data()

@st.cache_resource
def load_ml_assets():
    assets = {}
    assets["le"]    = joblib.load(MODEL_DIR / "label_encoder.pkl")
    assets["bow"]   = joblib.load(MODEL_DIR / "bow_vectorizer.pkl")
    assets["tfidf"] = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    assets["nb_bow"]   = joblib.load(MODEL_DIR / "nb_bow.pkl")
    assets["nb_tfidf"] = joblib.load(MODEL_DIR / "nb_tfidf.pkl")
    assets["lr_bow"]   = joblib.load(MODEL_DIR / "lr_bow.pkl")
    assets["lr_tfidf"] = joblib.load(MODEL_DIR / "lr_tfidf.pkl")

    if not TRANSFORMERS_AVAILABLE:
        assets["bert_available"] = False
        assets["bert_error"] = (
            "The `transformers` and/or `torch` packages are not installed.\n\n"
            "Install them with:\n```bash\npip install torch transformers\n```"
        )
    elif not BERT_MODEL_PATH.exists():
        assets["bert_available"] = False
        assets["bert_error"] = (
            f"BERT model directory not found at:\n`{BERT_MODEL_PATH}`\n\n"
            "Save your trained model with:\n```python\n"
            f"model.save_pretrained(r'{BERT_MODEL_PATH}')\n"
            f"tokenizer.save_pretrained(r'{BERT_MODEL_PATH}')\n```"
        )
    else:
        try:
            assets["bert_tokenizer"] = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
            assets["bert_model"] = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
            assets["bert_model"].eval()
            assets["bert_available"] = True
        except Exception as e:
            assets["bert_available"] = False
            assets["bert_error"] = f"BERT model found but failed to load.\n\n**Error:** `{e}`"

    return assets

assets = load_ml_assets()
le = assets["le"]

def make_wordcloud_img(text_series):
    combined = " ".join(text_series.dropna().astype(str).tolist())
    wc = WordCloud(
        width=900, height=420,
        background_color=None,
        mode="RGBA",
        colormap="cool",
        max_words=200,
        collocations=False,
    ).generate(combined)
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf

def get_top_words_per_class(df, le, vectorizer, n=10):
    results = {}
    for cls in le.classes_:
        subset = df[df["sentiment"] == cls]["review_description"]
        if len(subset) == 0:
            continue
        vec = CountVectorizer(max_features=n, stop_words="english")
        mat = vec.fit_transform(subset)
        counts = mat.sum(axis=0).A1
        words = vec.get_feature_names_out()
        results[cls] = pd.DataFrame({"word": words, "count": counts}).sort_values("count", ascending=False)
    return results

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠  Home",
    "🔬  Text Analyzer",
    "📂  Data Explorer",
    "📊  Visualizations",
    "🧩  Model Info",
    "⚗️  BERT (Experimental)",
])

with tab1:
    st.markdown("""
    <div style="padding: 48px 0 32px;">
        <div class="section-label">Natural Language Processing · Sentiment Analysis</div>
        <h1 style="font-size:3rem;font-weight:700;color:#fff;margin:8px 0 16px;line-height:1.1;">
            Syntax
        </h1>
        <p style="font-size:1.15rem;color:#9090b0;max-width:620px;line-height:1.7;margin-bottom:40px;">
            Uncover how Indonesian users feel about Indoenesian apps on the Play Store and App Store :
            classified as Positive, Neutral, or Negative using classical ML and transformer models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    total = len(df)
    pos_pct = round(len(df[df["sentiment"] == "Positive"]) / total * 100, 1) if total else 0
    neg_pct = round(len(df[df["sentiment"] == "Negative"]) / total * 100, 1) if total else 0
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [f"{total:,}", f"{pos_pct}%", f"{neg_pct}%", "5"],
        ["Total Reviews", "Positive Rate", "Negative Rate", "Models Trained (Classical + BERT)"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <h2 style="color:#fff;font-size:1.35rem;font-weight:700;margin-bottom:6px;">What Problem This Solves</h2>
    <p style="color:#9090b0;max-width:700px;line-height:1.7;margin-bottom:28px;">
        App developers receive thousands of reviews but rarely have time to read them all.
        Syntax automates opinion mining on Indonesian-language reviews, helping product teams
        identify user pain points and satisfaction drivers at scale; without manual reading.
    </p>
    <h2 style="color:#fff;font-size:1.35rem;font-weight:700;margin-bottom:16px;">How to Use the App</h2>
    """, unsafe_allow_html=True)

    g1, g2, g3, g4, g5 = st.columns(5)
    for col, num, title, desc in zip(
        [g1, g2, g3, g4, g5],
        ["01", "02", "03", "04", "05"],
        ["Text Analyzer", "Data Explorer", "Visualizations", "Model Info", "BERT (Exp.)"],
        [
            "Paste any translated Indonesian app review and choose a model to get an instant sentiment prediction with confidence scores and word-level influence.",
            "Browse sample rows from the dataset, check key statistics, and explore the label distribution before diving into charts.",
            "Eight charts covering word clouds, class balance, confusion matrices, model comparison, and temporal trends.",
            "Read descriptions of each model, compare performance metrics side-by-side, and understand what was used during training.",
            "Try the fine-tuned BERT transformer model for higher-accuracy predictions if the model files are loaded.",
        ],
    ):
        col.markdown(f"""
        <div class="model-card">
            <div class="section-label">{num}</div>
            <h4 style="margin-top:4px;">{title}</h4>
            <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <h2 style="color:#fff;font-size:1.35rem;font-weight:700;margin-bottom:20px;">Team Members</h2>
    """, unsafe_allow_html=True)

    team = [
        ("Muhammad Haziem Azfar bin Mohd Ransaimon", "Data Collection & Preprocessing"),
        ("Fadzlee Adam Bin Mohd Nazlee", "Feature Engineering & Classical ML"),
        ("Muhammad Farkhin bin Abd Baser", "BERT Fine-tuning & Evaluation"),
        ("Mohamad Afiq Ikmal Bin Mohd Roslan", "Streamlit App & Visualisations"),
    ]
    cols = st.columns(len(team))
    for col, (name, role) in zip(cols, team):
        col.markdown(f"""
        <div class="model-card" style="text-align:center;">
            <div style="font-size:2.2rem;margin-bottom:8px;">👤</div>
            <h4>{name}</h4>
            <p>{role}</p>
        </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <h2 style="color:#fff;font-size:1.5rem;font-weight:700;margin-bottom:4px;">Text Analyzer</h2>
    <p style="color:#9090b0;margin-bottom:28px;">Enter an Indonesian app review below and select a model configuration to classify its sentiment.</p>
    """, unsafe_allow_html=True)

    col_clf, col_feat = st.columns(2)
    with col_clf:
        classifier_choice = st.selectbox("Classifier", ["Logistic Regression", "Naive Bayes"], key="pred_clf")
    with col_feat:
        feature_choice = st.selectbox("Feature Representation", ["TF-IDF", "BoW"], key="pred_feat")

    clf_prefix  = "lr" if classifier_choice == "Logistic Regression" else "nb"
    feat_suffix = "bow" if feature_choice == "BoW" else "tfidf"
    active_model      = assets[f"{clf_prefix}_{feat_suffix}"]
    active_vectorizer = assets[feat_suffix]

    review_text = st.text_area(
        "Review Text",
        placeholder="Paste or type an Indonesian app review here…",
        height=140,
        key="analyze_text",
    )

    if st.button("Analyze Sentiment"):
        if not review_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            transformed = active_vectorizer.transform([review_text])
            probs       = active_model.predict_proba(transformed)[0]
            pred_idx    = active_model.predict(transformed)[0]
            sentiment   = le.inverse_transform([pred_idx])[0]

            badge_color = SENTIMENT_COLORS.get(sentiment, "#a78bfa")
            st.markdown(f"""
            <div style="margin:24px 0 28px;">
                <div class="section-label" style="margin-bottom:8px;">Prediction</div>
                <span class="prediction-badge" style="background:{badge_color}22;color:{badge_color};border:1.5px solid {badge_color}44;">
                    {sentiment}
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-bottom:12px;">Confidence Scores</div>', unsafe_allow_html=True)
            classes = list(le.classes_)
            for idx, cls in enumerate(classes):
                score = float(probs[idx])
                col_bar, col_val = st.columns([5, 1])
                col_bar.markdown(f"<span style='color:#9090b0;font-size:0.85rem;'>{cls}</span>", unsafe_allow_html=True)
                col_bar.progress(score)
                col_val.markdown(f"<span style='color:#d4d4e8;font-size:0.9rem;font-family:JetBrains Mono,monospace;'>{score*100:.1f}%</span>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-label" style="margin-bottom:12px;">Word Influence (Top Features)</div>', unsafe_allow_html=True)

            try:
                feature_names = active_vectorizer.get_feature_names_out()
                input_vec     = transformed.toarray()[0]
                nonzero_idx   = np.where(input_vec > 0)[0]

                if hasattr(active_model, "coef_"):
                    coefs       = active_model.coef_[pred_idx]
                    word_scores = [(feature_names[i], float(coefs[i] * input_vec[i])) for i in nonzero_idx]
                elif hasattr(active_model, "feature_log_prob_"):
                    log_probs   = active_model.feature_log_prob_[pred_idx]
                    word_scores = [(feature_names[i], float(log_probs[i])) for i in nonzero_idx]
                else:
                    word_scores = []

                if word_scores:
                    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_n = word_scores[:15]
                    wdf   = pd.DataFrame(top_n, columns=["Word", "Influence"])

                    fig_inf = px.bar(
                        wdf, x="Influence", y="Word", orientation="h",
                        color="Influence",
                        color_continuous_scale=["#f87171", "#a78bfa", "#34d399"],
                    )
                    fig_inf.update_layout(
                        **PLOT_THEME,
                        coloraxis_showscale=False,
                        height=380,
                        margin=dict(l=0, r=0, t=10, b=10),
                    )
                    st.plotly_chart(fig_inf, use_container_width=True)
            except Exception as e:
                st.info(f"Word influence chart unavailable: {e}")

with tab3:
    st.markdown("""
    <h2 style="color:#fff;font-size:1.5rem;font-weight:700;margin-bottom:4px;">Data Explorer</h2>
    <p style="color:#9090b0;margin-bottom:28px;">Browse the dataset, review its statistics, and check label distribution at a glance.</p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-bottom:10px;">Sample Rows</div>', unsafe_allow_html=True)
    n_sample = st.slider("Rows to display", 5, 100, 20, key="sample_slider")
    display_cols = [c for c in ["review_description", "sentiment", "rating", "source", "review_date", "text_length"] if c in df.columns]
    st.dataframe(df[display_cols].sample(n=min(n_sample, len(df)), random_state=42).reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:10px;">Dataset Statistics</div>', unsafe_allow_html=True)

    stat_cols = st.columns(4)
    stats = [
        ("Total Records", f"{len(df):,}"),
        ("Unique Sentiments", str(df["sentiment"].nunique())),
        ("Avg Review Length", f"{df['text_length'].mean():.0f} words"),
        ("Longest Review", f"{df['text_length'].max()} words"),
    ]
    for col, (lbl, val) in zip(stat_cols, stats):
        col.markdown(f"""
        <div class="metric-card"><div class="value" style="font-size:1.5rem;">{val}</div><div class="label">{lbl}</div></div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df[display_cols].describe(include="all").fillna("—"), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:10px;">Label Distribution</div>', unsafe_allow_html=True)

    dist_c1, dist_c2 = st.columns(2)
    sc = df["sentiment"].value_counts().reset_index()
    sc.columns = ["Sentiment", "Count"]
    with dist_c1:
        fig_pie = px.pie(
            sc, values="Count", names="Sentiment",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.45,
        )
        fig_pie.update_layout(**PLOT_THEME, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    with dist_c2:
        fig_bar = px.bar(
            sc, x="Sentiment", y="Count",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(**PLOT_THEME, showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    st.markdown("""
    <h2 style="color:#fff;font-size:1.5rem;font-weight:700;margin-bottom:4px;">Visualizations</h2>
    <p style="color:#9090b0;margin-bottom:28px;">Eight charts covering word distributions, model performance, and temporal patterns.</p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-bottom:10px;">① Word Cloud — Most Frequent Terms</div>', unsafe_allow_html=True)
    if WORDCLOUD_AVAILABLE:
        try:
            wc_buf = make_wordcloud_img(df["review_description"])
            st.image(wc_buf, use_container_width=True)
        except Exception as e:
            st.info(f"Word cloud generation failed: {e}")
    else:
        st.info("Install `wordcloud` to enable this chart: `pip install wordcloud`")

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:10px;">② Class Distribution (Bar + Pie)</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    sc = df["sentiment"].value_counts().reset_index()
    sc.columns = ["Sentiment", "Count"]
    with v1:
        fig_b2 = px.bar(sc, x="Sentiment", y="Count", color="Sentiment", color_discrete_map=SENTIMENT_COLORS, text="Count")
        fig_b2.update_traces(textposition="outside")
        fig_b2.update_layout(**PLOT_THEME, showlegend=False, title="Label Count")
        st.plotly_chart(fig_b2, use_container_width=True)
    with v2:
        fig_p2 = px.pie(sc, values="Count", names="Sentiment", color="Sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.4)
        fig_p2.update_layout(**PLOT_THEME, title="Label Share")
        st.plotly_chart(fig_p2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:10px;">③ Confusion Matrix Heatmap</div>', unsafe_allow_html=True)

    cm_configs = {
        "Logistic Regression (TF-IDF)": {"model": assets["lr_tfidf"], "vec": assets["tfidf"]},
        "Logistic Regression (BoW)":    {"model": assets["lr_bow"],   "vec": assets["bow"]},
        "Naive Bayes (TF-IDF)":         {"model": assets["nb_tfidf"], "vec": assets["tfidf"]},
        "Naive Bayes (BoW)":            {"model": assets["nb_bow"],   "vec": assets["bow"]},
    }
    cm_sel = st.selectbox("Select model for confusion matrix", list(cm_configs.keys()), key="cm_sel")
    X_raw  = df["review_description"]
    y_true = le.transform(df["sentiment"])
    sel    = cm_configs[cm_sel]
    y_pred = sel["model"].predict(sel["vec"].transform(X_raw))
    cm     = confusion_matrix(y_true, y_pred)
    labels = list(le.classes_)

    fig_cm = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale=[[0, "#0f0f22"], [0.5, "#4c1d95"], [1, "#a78bfa"]],
        text_auto=True,
        aspect="auto",
    )
    fig_cm.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d4d4e8", "family": "Inter"},
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
        coloraxis_showscale=False,
    )
    fig_cm.update_xaxes(gridcolor="#1e1e38", linecolor="#2a2a50", tickfont={"color": "#9090b0"})
    fig_cm.update_yaxes(gridcolor="#1e1e38", linecolor="#2a2a50", tickfont={"color": "#9090b0"})
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:10px;">④ Model Comparison — Accuracy & F1</div>', unsafe_allow_html=True)

    results = []
    for name, comp in cm_configs.items():
        Xt = comp["vec"].transform(X_raw)
        yp = comp["model"].predict(Xt)
        results.append({
            "Model": name,
            "Accuracy":  round(accuracy_score(y_true, yp), 4),
            "Precision": round(precision_score(y_true, yp, average="weighted", zero_division=0), 4),
            "Recall":    round(recall_score(y_true, yp, average="weighted", zero_division=0), 4),
            "F1-Score":  round(f1_score(y_true, yp, average="weighted", zero_division=0), 4),
        })
    res_df = pd.DataFrame(results)

    metrics_melt = res_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )
    fig_comp = px.bar(
        metrics_melt, x="Model", y="Score", color="Metric",
        barmode="group",
        color_discrete_sequence=["#a78bfa", "#34d399", "#f87171", "#fbbf24"],
        text_auto=".3f",
    )
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(**PLOT_THEME, height=420, margin=dict(t=20, b=20))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:10px;">⑤ Top 20 Words (Global)</div>', unsafe_allow_html=True)
    try:
        cv20  = CountVectorizer(stop_words="english", max_features=20)
        wm20  = cv20.fit_transform(df["review_description"])
        wc20  = wm20.sum(axis=0).A1
        w20   = cv20.get_feature_names_out()
        top20 = pd.DataFrame({"Word": w20, "Frequency": wc20}).sort_values("Frequency")
        fig_top = px.bar(
            top20, x="Frequency", y="Word", orientation="h",
            color="Frequency",
            color_continuous_scale=["#4c1d95", "#a78bfa", "#ddd6fe"],
        )
        fig_top.update_layout(**PLOT_THEME, coloraxis_showscale=False, height=480, margin=dict(l=0, r=0))
        st.plotly_chart(fig_top, use_container_width=True)
    except Exception as e:
        st.error(f"Top words chart failed: {e}")

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:10px;">⑥ Text Length Distribution by Sentiment</div>', unsafe_allow_html=True)
    fig_len = px.violin(
        df[df["text_length"] < df["text_length"].quantile(0.99)],
        x="sentiment", y="text_length",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        box=True,
        points="outliers",
    )
    fig_len.update_layout(**PLOT_THEME, showlegend=False, xaxis_title="Sentiment", yaxis_title="Word Count")
    st.plotly_chart(fig_len, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:10px;">⑦ Sentiment Trend Over Time</div>', unsafe_allow_html=True)
    if "review_date" in df.columns:
        df_t = df.copy()
        df_t["review_date"] = pd.to_datetime(df_t["review_date"], errors="coerce")
        df_t = df_t.dropna(subset=["review_date"])
        if len(df_t) > 0:
            monthly = (
                df_t.groupby([pd.Grouper(key="review_date", freq="ME"), "sentiment"])
                .size().unstack(fill_value=0).reset_index()
            )
            fig_trend = go.Figure()
            for cls in ["Positive", "Neutral", "Negative"]:
                if cls in monthly.columns:
                    fig_trend.add_trace(go.Scatter(
                        x=monthly["review_date"], y=monthly[cls],
                        mode="lines+markers", name=cls,
                        line=dict(color=SENTIMENT_COLORS[cls], width=2.5),
                        marker=dict(size=5),
                    ))
            fig_trend.update_layout(**PLOT_THEME, xaxis_title="Month", yaxis_title="Review Count")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No valid date data available for this chart.")
    else:
        st.info("No `review_date` column found in the dataset.")

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:10px;">⑧ Top Words per Sentiment Class</div>', unsafe_allow_html=True)
    try:
        per_class = get_top_words_per_class(df, le, assets["tfidf"], n=10)
        pw_cols   = st.columns(len(per_class))
        for col, (cls, wdf) in zip(pw_cols, per_class.items()):
            fig_pw = px.bar(
                wdf, x="count", y="word", orientation="h",
                color_discrete_sequence=[SENTIMENT_COLORS.get(cls, "#a78bfa")],
                title=cls,
            )
            fig_pw.update_layout(**PLOT_THEME, showlegend=False, height=320, margin=dict(l=0, r=0, t=36, b=0))
            col.plotly_chart(fig_pw, use_container_width=True)
    except Exception as e:
        st.error(f"Per-class word chart failed: {e}")

with tab5:
    st.markdown("""
    <h2 style="color:#fff;font-size:1.5rem;font-weight:700;margin-bottom:4px;">Model Info</h2>
    <p style="color:#9090b0;margin-bottom:28px;">Architecture descriptions, training details, and performance benchmarks for every model.</p>
    """, unsafe_allow_html=True)

    model_descriptions = [
        {
            "name": "Logistic Regression + TF-IDF",
            "key": "lr_tfidf",
            "vec_key": "tfidf",
            "description": (
                "TF-IDF (Term Frequency–Inverse Document Frequency) down-weights words that appear "
                "across many documents and highlights discriminative terms. Paired with L2-regularised "
                "Logistic Regression, this is typically the strongest classical baseline for short text "
                "classification tasks."
            ),
            "training": (
                "Trained on balanced_app_reviews.csv with an 80/20 train–test split. "
                "TF-IDF fitted on training set only; max_features tuned via cross-validation."
            ),
        },
        {
            "name": "Logistic Regression + BoW",
            "key": "lr_bow",
            "vec_key": "bow",
            "description": (
                "Bag-of-Words converts text to raw term counts. Combined with Logistic Regression, "
                "it provides a fast interpretable baseline. Useful for shorter reviews where word presence "
                "matters more than relative frequency."
            ),
            "training": "Same split as above. CountVectorizer with unigrams, fitted on train set.",
        },
        {
            "name": "Naïve Bayes + TF-IDF",
            "key": "nb_tfidf",
            "vec_key": "tfidf",
            "description": (
                "Multinomial Naïve Bayes assumes conditional independence between features. It is "
                "computationally very efficient and often surprisingly competitive on text tasks, "
                "especially when training data is limited."
            ),
            "training": (
                "Alpha (Laplace smoothing) tuned via grid search. TF-IDF vectors must be "
                "non-negative (ensured by using sublinear_tf=False)."
            ),
        },
        {
            "name": "Naïve Bayes + BoW",
            "key": "nb_bow",
            "vec_key": "bow",
            "description": (
                "The simplest combination — raw counts fed into Naïve Bayes. Serves as the floor "
                "baseline. Fast to train and predict, interpretable through class log-probabilities."
            ),
            "training": "Alpha=1.0 (default Laplace smoothing). No feature selection applied.",
        },
    ]

    X_raw  = df["review_description"]
    y_true = le.transform(df["sentiment"])

    for md in model_descriptions:
        with st.expander(f"📦  {md['name']}", expanded=False):
            st.markdown(f"""
            <div class="model-card">
                <p style="margin-bottom:14px;">{md['description']}</p>
                <div class="section-label" style="margin-bottom:4px;">Training Details</div>
                <p style="font-size:0.83rem;">{md['training']}</p>
            </div>
            """, unsafe_allow_html=True)

            model = assets[md["key"]]
            vec   = assets[md["vec_key"]]
            Xt    = vec.transform(X_raw)
            yp    = model.predict(Xt)

            acc = accuracy_score(y_true, yp)
            pr  = precision_score(y_true, yp, average="weighted", zero_division=0)
            rc  = recall_score(y_true, yp, average="weighted", zero_division=0)
            f1  = f1_score(y_true, yp, average="weighted", zero_division=0)

            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, label, val in zip(
                [mc1, mc2, mc3, mc4],
                ["Accuracy", "Precision", "Recall", "F1-Score"],
                [acc, pr, rc, f1],
            ):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="value" style="font-size:1.4rem;">{val:.4f}</div>
                    <div class="label">{label}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="margin-bottom:12px;">All Models — Performance Summary</div>', unsafe_allow_html=True)

    rows = []
    for md in model_descriptions:
        Xt = assets[md["vec_key"]].transform(X_raw)
        yp = assets[md["key"]].predict(Xt)
        rows.append({
            "Model": md["name"],
            "Accuracy":  f"{accuracy_score(y_true, yp):.4f}",
            "Precision": f"{precision_score(y_true, yp, average='weighted', zero_division=0):.4f}",
            "Recall":    f"{recall_score(y_true, yp, average='weighted', zero_division=0):.4f}",
            "F1-Score":  f"{f1_score(y_true, yp, average='weighted', zero_division=0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

with tab6:
    st.markdown("""
    <h2 style="color:#fff;font-size:1.5rem;font-weight:700;margin-bottom:4px;">BERT — Experimental</h2>
    <p style="color:#9090b0;margin-bottom:28px;">
        Fine-tuned IndoBERT transformer model. Contextual embeddings outperform bag-of-words
        representations but require a pre-loaded model directory.
    </p>
    """, unsafe_allow_html=True)

    if not assets.get("bert_available", False):
        st.error("BERT model failed to load.")
        st.markdown(assets.get("bert_error", "Unknown error. Check the model path and try again."))
    else:
        bert_text = st.text_area(
            "Review Text (for BERT)",
            placeholder="Paste an Indonesian app review here…",
            height=140,
            key="bert_input_text",
        )
        if st.button("Run BERT Prediction", key="bert_predict_btn"):
            if not bert_text.strip():
                st.warning("Please enter some text first.")
            else:
                tokenizer  = assets["bert_tokenizer"]
                bert_model = assets["bert_model"]
                with st.spinner("Running tokenization and forward pass…"):
                    inputs = tokenizer(
                        bert_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    with torch.no_grad():
                        outputs       = bert_model(**inputs)
                        logits        = outputs.logits
                        probabilities = torch.softmax(logits, dim=1).flatten().tolist()
                        pred_id       = torch.argmax(logits, dim=1).item()

                try:
                    sentiment = le.inverse_transform([pred_id])[0]
                except Exception:
                    classes   = list(le.classes_)
                    sentiment = classes[pred_id] if pred_id < len(classes) else "Unknown"

                badge_color = SENTIMENT_COLORS.get(sentiment, "#a78bfa")
                st.markdown(f"""
                <div style="margin:24px 0 28px;">
                    <div class="section-label" style="margin-bottom:8px;">BERT Prediction</div>
                    <span class="prediction-badge" style="background:{badge_color}22;color:{badge_color};border:1.5px solid {badge_color}44;">
                        {sentiment}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                classes = list(le.classes_)
                for idx, cls in enumerate(classes):
                    if idx < len(probabilities):
                        score = float(probabilities[idx])
                        col_b, col_v = st.columns([5, 1])
                        col_b.markdown(f"<span style='color:#9090b0;font-size:0.85rem;'>{cls}</span>", unsafe_allow_html=True)
                        col_b.progress(score)
                        col_v.markdown(f"<span style='color:#d4d4e8;font-size:0.9rem;font-family:JetBrains Mono,monospace;'>{score*100:.1f}%</span>", unsafe_allow_html=True)