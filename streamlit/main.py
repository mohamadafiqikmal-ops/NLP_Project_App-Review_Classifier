import streamlit as st
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="App Review Classifier",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
  /* Black background */
  .stApp { background-color: #000000; }
  section[data-testid="stSidebar"] { background-color: #0a0a0a; }

  /* Text */
  html, body, [class*="css"], p, div, span, label { color: #e5e5e5; }

  /* Tab bar */
  .stTabs [data-baseweb="tab-list"] {
      background: #000;
      border-bottom: 1px solid #2a2a2a;
      gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent;
      color: #666;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 12px 28px;
      border: none;
      border-bottom: 2px solid transparent;
      margin-bottom: -1px;
  }
  .stTabs [aria-selected="true"] {
      color: #fff !important;
      border-bottom: 2px solid #fff !important;
      background: transparent !important;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #ccc !important; }

  /* Tab content area */
  .stTabs [data-baseweb="tab-panel"] {
      background: #000;
      padding-top: 32px;
  }

  /* Inputs, textareas, selects */
  .stTextInput input,
  .stTextArea textarea,
  .stSelectbox div[data-baseweb="select"] {
      background: #111 !important;
      border: 1px solid #333 !important;
      color: #e5e5e5 !important;
      border-radius: 8px !important;
  }

  /* Buttons */
  .stButton button {
      background: #000080;
      color: #fff;
      border: none;
      font-weight: 600;
      border-radius: 8px;
  }
  .stButton button:hover { background: #0000b3; }

  /* Dividers */
  hr { border-color: #222 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #000; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="color:#fff;font-size:1.8rem;font-weight:700;margin-bottom:4px;">
  Indonesian App Review Sentiment Analysis
</h1>
<p style="color:#555;font-size:0.9rem;margin-bottom:32px;">
  An AI capable of executing sentiment analysis based on Indonesian app reviews!
</p>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = PROJECT_DIR / "notebooks" / "balanced_app_reviews.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    
    if "translation" in df.columns:
        df = df.rename(columns={"translation": "review_description"})
    elif "clean_content" in df.columns:
        df = df.rename(columns={"clean_content": "review_description"})
    elif "content" in df.columns:
        df = df.rename(columns={"content": "review_description"})

    if "labelScore" in df.columns:
        df = df.rename(columns={"labelScore": "sentiment"})

    if "score" in df.columns:
        df = df.rename(columns={"score": "rating"})

    if "at" in df.columns:
        df = df.rename(columns={"at": "review_date"})
        
    if "app" in df.columns:
        df = df.rename(columns={"app": "source"})

    df["review_description"] = df["review_description"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip().str.capitalize()
    df.loc[df["sentiment"] == "", "sentiment"] = "Neutral"
    
    return df

df = load_data()

@st.cache_resource
def load_ml_assets():
    assets = {}
    
    assets["le"] = joblib.load(MODEL_DIR / "label_encoder.pkl")
    assets["bow"] = joblib.load(MODEL_DIR / "bow_vectorizer.pkl")
    assets["tfidf"] = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    
    # 2. Load the 4 model permutations
    assets["nb_bow"] = joblib.load(MODEL_DIR / "nb_bow.pkl")
    assets["nb_tfidf"] = joblib.load(MODEL_DIR / "nb_tfidf.pkl")
    assets["lr_bow"] = joblib.load(MODEL_DIR / "lr_bow.pkl")
    assets["lr_tfidf"] = joblib.load(MODEL_DIR / "lr_tfidf.pkl")
    
    try:
        bert_path = MODEL_DIR / "bert_model"
        if bert_path.exists():
            assets["bert_tokenizer"] = AutoTokenizer.from_pretrained(bert_path)
            assets["bert_model"] = AutoModelForSequenceClassification.from_pretrained(bert_path)
        else:
            assets["bert_tokenizer"] = AutoTokenizer.from_pretrained("bert-base-uncased")
            assets["bert_model"] = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        assets["bert_model"].eval()
        assets["bert_available"] = True
    except Exception as e:
        assets["bert_available"] = False
        
    return assets

assets = load_ml_assets()
le = assets["le"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮  Predict",
    "📊  Visualize",
    "⚖️  Model Comparison",
    "👋  About",
    "Experimental (BERT)",
])

with tab1:
    st.subheader("🧠 Model Configuration Matrix")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        classifier_choice = st.selectbox("Select Classifier Algorithm", ["Logistic Regression", "Naive Bayes"], key="pred_clf")
    with col_m2:
        feature_choice = st.selectbox("Select Text Feature Representation Strategy", ["BoW", "TF-IDF"], key="pred_feat")
    
    clf_prefix = "lr" if classifier_choice == "Logistic Regression" else "nb"
    feat_suffix = "bow" if feature_choice == "BoW" else "tfidf"
    
    active_model = assets[f"{clf_prefix}_{feat_suffix}"]
    active_vectorizer = assets[feat_suffix]
    
    text = st.text_input("Enter review string for classification evaluation:")

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter valid text to execute a prediction sequence.")
        else:
            transformed_input = active_vectorizer.transform([text])
            
            conf = active_model.predict_proba(transformed_input)[0]
            pred_numeric = active_model.predict(transformed_input)
            sentiment = le.inverse_transform(pred_numeric)[0]
            
            st.markdown(f"### Predicted Result Class Strategy: **{sentiment}**")
            
            classes = list(le.classes_)
            for index, class_label in enumerate(classes):
                st.write(f"### {class_label}")
                c1, c2 = st.columns([3, 1])
                score = float(conf[index])
                with c1:
                    st.progress(score)
                with c2:
                    st.write(f"{score*100:.1f}%")

with tab2:
    st.subheader("📊 Advanced Dataset Analytics & Visualizations")
    st.markdown("---")

    plotly_layout_theme = {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#e5e5e5'},
        'xaxis': {'gridcolor': '#222', 'linecolor': '#333'},
        'yaxis': {'gridcolor': '#222', 'linecolor': '#333'}
    }

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.write("### 🌌 Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        
        color_map = {"Positive": "#00CC96", "Neutral": "#FECB52", "Negative": "#EF553B"}
        
        fig_sent = px.pie(
            sentiment_counts, 
            values="Count", 
            names="Sentiment",
            color="Sentiment",
            color_discrete_map=color_map,
            hole=0.4
        )
        fig_sent.update_layout(**plotly_layout_theme)
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_v2:
        st.write("### 🌟 Rating Breakdown")
        if "rating" in df.columns:
            rating_counts = df["rating"].value_counts().reset_index()
            rating_counts.columns = ["Rating", "Count"]
            rating_counts = rating_counts.sort_values(by="Rating")
            
            fig_rate = px.bar(
                rating_counts, 
                x="Rating", 
                y="Count",
                text="Count",
                color_discrete_sequence=["#000080"]
            )
            fig_rate.update_layout(**plotly_layout_theme)
            fig_rate.update_traces(textposition='outside', marker_line_color='#333', marker_line_width=1.5)
            st.plotly_chart(fig_rate, use_container_width=True)

    st.markdown("---")
    col_v3, col_v4 = st.columns([1, 2])

    with col_v3:
        st.write("### 📱 Platform Distribution")
        if "source" in df.columns:
            source_counts = df["source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Count"]
            
            fig_source = px.bar(
                source_counts, 
                y="Source", 
                x="Count", 
                orientation='h',
                color_discrete_sequence=["#4b0082"]
            )
            fig_source.update_layout(**plotly_layout_theme)
            st.plotly_chart(fig_source, use_container_width=True)

    with col_v4:
        st.write("### 📈 Sentiment Trend Over Time")
        if "review_date" in df.columns:
            df_trend = df.copy()
            df_trend["review_date"] = pd.to_datetime(df_trend["review_date"], errors="coerce")
            df_trend = df_trend.dropna(subset=["review_date"])

            if len(df_trend) > 0:
                monthly_sentiment = (
                    df_trend
                    .groupby([pd.Grouper(key="review_date", freq="ME"), "sentiment"])
                    .size()
                    .unstack(fill_value=0)
                    .reset_index()
                )
                
                fig_trend = go.Figure()
                for sentiment_class in ["Positive", "Neutral", "Negative"]:
                    if sentiment_class in monthly_sentiment.columns:
                        fig_trend.add_trace(go.Scatter(
                            x=monthly_sentiment["review_date"],
                            y=monthly_sentiment[sentiment_class],
                            mode='lines+markers',
                            name=sentiment_class,
                            line=dict(color=color_map.get(sentiment_class, "#fff"), width=3)
                        ))
                
                fig_trend.update_layout(**plotly_layout_theme)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.warning("No valid review dates discovered to project charts timeline.")

    st.markdown("---")
    st.write("### 🔤 Top 20 Frequently Occurring Keyword Contexts")
    try:
        vectorizer = CountVectorizer(stop_words="english", max_features=20)
        word_matrix = vectorizer.fit_transform(df["review_description"])
        word_counts = word_matrix.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()

        top_words_df = pd.DataFrame({
            "Word": words,
            "Frequency Count": word_counts
        }).sort_values(by="Frequency Count", ascending=True)

        fig_words = px.bar(
            top_words_df, 
            x="Frequency Count", 
            y="Word", 
            orientation='h',
            color="Frequency Count",
            color_continuous_scale="Viridis"
        )
        fig_words.update_layout(**plotly_layout_theme)
        fig_words.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_words, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to generate top words visualization: {e}")

with tab3:
    st.subheader("Comprehensive Cross-Configuration Framework Evaluation")
    
    X_raw = df["review_description"]
    y_true = le.transform(df["sentiment"])
    
    configurations = {
        "Logistic Regression (BoW)": {"model": assets["lr_bow"], "vec": assets["bow"]},
        "Logistic Regression (TF-IDF)": {"model": assets["lr_tfidf"], "vec": assets["tfidf"]},
        "Naive Bayes (BoW)": {"model": assets["nb_bow"], "vec": assets["bow"]},
        "Naive Bayes (TF-IDF)": {"model": assets["nb_tfidf"], "vec": assets["tfidf"]},
    }
    
    results = []
    for config_name, components in configurations.items():
        X_transformed = components["vec"].transform(X_raw)
        y_pred = components["model"].predict(X_transformed)
        
        results.append({
            "Configuration Strategy": config_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        })
        
    results_df = pd.DataFrame(results)
    st.write("### Production Framework Metrics Log Summary")
    st.dataframe(results_df.set_index("Configuration Strategy"))
    
    st.write("### Pipeline Configuration Accuracy Scores")
    st.bar_chart(results_df.set_index("Configuration Strategy")[["Accuracy"]])

    st.write("### Confusion Matrix Inspection Configuration")
    cm_choice = st.selectbox(
        "Choose variant structure for Confusion Matrix evaluation",
        list(configurations.keys()),
        key="cm_select"
    )
    
    selected_components = configurations[cm_choice]
    X_test_transformed = selected_components["vec"].transform(X_raw)
    y_test_pred = selected_components["model"].predict(X_test_transformed)
    
    cm = confusion_matrix(y_true, y_test_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    st.dataframe(cm_df)

with tab4:
    st.subheader("About This Project")
    st.write("""
    This Natural Language Processing application performs classification workflows for app review sentiment tracking.
    Reviews are systematically analyzed and grouped into positive, neutral, or negative classes.
    """)
    st.write("### Project Theme: Theme 6: App Review Classifier")
    st.write("### Framework Processing Pipeline Structure")
    st.write("""
    1. Load input operational textual evaluation datasets.  
    2. Normalize target metrics values utilizing feature extractions via Bag-of-Words and TF-IDF mappings.  
    3. Run classification matrix selections natively across isolated processing configurations.
    """)

with tab5:
    st.subheader("BERT: Bidirectional Encoder Representations from Transformers")
    
    if not assets.get("bert_available", False):
        st.error("❌ The underlying BERT neural architecture is missing or failed to initialize properly.")
        st.info("Ensure your transformers checkpoint weights are saved inside `./streamlit/models/bert_model/` configuration paths.")
    else:
        
        bert_text = st.text_area("Enter review string for Transformer contextual classification analysis:", key="bert_input_text")
        
        if st.button("Predict", key="bert_predict_btn"):
            if not bert_text.strip():
                st.warning("Please provide structural context input to pass downstream to the Transformer tokenization matrix.")
            else:
                tokenizer = assets["bert_tokenizer"]
                model = assets["bert_model"]
                
                with st.spinner("Executing sequence classification calculations across neural tensor maps..."):
                    inputs = tokenizer(
                        bert_text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512
                    )
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        
                        probabilities = torch.softmax(logits, dim=1).flatten().tolist()
                        prediction_id = torch.argmax(logits, dim=1).item()
                
                try:
                    predicted_sentiment = le.inverse_transform([prediction_id])[0]
                except Exception:
                    fallback_classes = list(le.classes_)
                    predicted_sentiment = fallback_classes[prediction_id] if prediction_id < len(fallback_classes) else "Unknown"
                
                st.markdown(f"### Prediction: **{predicted_sentiment}**")
                
                classes = list(le.classes_)
                for index, class_label in enumerate(classes):
                    if index < len(probabilities):
                        st.write(f"### {class_label}")
                        c1, c2 = st.columns([3, 1])
                        score = float(probabilities[index])
                        with c1:
                            st.progress(score)
                        with c2:
                            st.write(f"{score*100:.1f}%")
                            
        