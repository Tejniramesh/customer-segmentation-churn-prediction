import os
import io
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Churn Prediction using AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------------
APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent
BASE_DIR = APP_DIR.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

BERT_MODEL_DIR = MODELS_DIR / "bert_final"
RAG_MODEL_DIR = MODELS_DIR / "rag"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_purchase_model.keras"
SEGMENTS_CSV = BASE_DIR / "customer_segments.csv"

IMG_SEGMENT_DIST = OUTPUTS_DIR / "segment_distribution.png"
IMG_CHURN_BY_SEGMENT = OUTPUTS_DIR / "churn_rate_by_segment.png"
IMG_ELBOW = OUTPUTS_DIR / "elbow_curve.png"
IMG_FEATURE_IMPORTANCE = OUTPUTS_DIR / "feature_importance.png"
IMG_LSTM_LOSS = OUTPUTS_DIR / "lstm_loss.png"
IMG_ROC_CURVE = OUTPUTS_DIR / "roc_curve.png"
IMG_BERT_CM = OUTPUTS_DIR / "bert_confusion_matrix.png"


def resolve_path(path):
    """
    Resolve a path robustly. If the given path doesn't exist, fall back to
    searching common alternate locations (e.g. relative to the current working
    directory, or one level up) so images load correctly regardless of where
    `streamlit run` was launched from.
    """
    path = Path(path)
    if path.exists():
        return path

    candidates = [
        BASE_DIR / path.name if path.parent.name else path,
        BASE_DIR / "outputs" / path.name,
        APP_DIR / "outputs" / path.name,
        Path.cwd() / "outputs" / path.name,
        Path.cwd() / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path

# ----------------------------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }

    .main {
        background-color: #f5f7fb;
    }

    /* Gradient Header */
    .app-header {
        background: linear-gradient(120deg, #4b6cb7 0%, #182848 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(24, 40, 72, 0.25);
    }
    .app-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: white;
    }
    .app-header p {
        font-size: 1.02rem;
        opacity: 0.92;
        margin: 0;
        color: #e8ecf7;
    }

    /* KPI / Metric Cards */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border: 1px solid #eef0f5;
        text-align: left;
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 26px rgba(0,0,0,0.10);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #182848;
        margin-bottom: 0.2rem;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #9099a8;
    }

    /* Feature / Info Cards */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        border: 1px solid #eef0f5;
        height: 100%;
        margin-bottom: 1rem;
    }
    .info-card h3 {
        margin-top: 0;
        color: #182848;
        font-size: 1.15rem;
    }
    .info-card p {
        color: #555b66;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3b4cca;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #182848;
        margin: 1.2rem 0 0.8rem 0;
        border-left: 5px solid #4b6cb7;
        padding-left: 0.7rem;
    }

    .workflow-step {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        border: 1px solid #eef0f5;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        text-align: center;
        height: 100%;
    }
    .workflow-step .step-num {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        border-radius: 50%;
        background: #4b6cb7;
        color: white;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .footer-box {
        text-align: center;
        padding: 1.2rem;
        margin-top: 2.5rem;
        color: #8b91a0;
        font-size: 0.85rem;
        border-top: 1px solid #e2e5ec;
    }

    .result-box {
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        font-weight: 700;
        font-size: 1.15rem;
        margin-top: 1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }
    .result-positive { background: #e7f8ee; color: #17803d; border: 1px solid #b6ecc8; }
    .result-negative { background: #fdeaea; color: #c62828; border: 1px solid #f6c1c1; }
    .result-neutral  { background: #fff8e1; color: #a16207; border: 1px solid #fde8a1; }

    .context-chunk {
        background: #f8f9fc;
        border-left: 4px solid #4b6cb7;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
        color: #3a3f4b;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #182848 0%, #223a6b 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #f0f2f8 !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------
def render_header(title, subtitle):
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title, text):
    st.markdown(
        f"""
        <div class="info-card">
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_image_if_exists(path, caption):
    resolved = resolve_path(path)
    if resolved.exists():
        st.image(str(resolved), caption=caption, use_container_width=True)
    else:
        try:
            display_path = os.path.relpath(resolved, BASE_DIR)
        except ValueError:
            display_path = str(resolved)
        st.info(f"Image not found: `{display_path}`")


@st.cache_data(show_spinner=False)
def load_segments_csv():
    if os.path.exists(SEGMENTS_CSV):
        try:
            return pd.read_csv(SEGMENTS_CSV)
        except Exception as e:
            st.error(f"Failed to read customer_segments.csv: {e}")
            return None
    return None


def render_footer():
    st.markdown(
        """
        <div class="footer-box">
            🧠 Customer Segmentation & Churn Prediction using AI &nbsp;|&nbsp;
            Built with Streamlit, TensorFlow, Transformers &amp; FAISS &nbsp;|&nbsp;
            © 2026 All Rights Reserved
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------------
# CACHED MODEL LOADERS
# ----------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading BERT sentiment model...")
def load_bert_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    if not os.path.exists(BERT_MODEL_DIR):
        return None, None, None
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.eval()
    return tokenizer, model, torch


@st.cache_resource(show_spinner="Loading LSTM purchase prediction model...")
def load_lstm_model():
    if not os.path.exists(LSTM_MODEL_PATH):
        return None
    from tensorflow import keras
    model = keras.models.load_model(LSTM_MODEL_PATH)
    return model


@st.cache_resource(show_spinner="Loading FAISS knowledge base...")
def load_rag_index():
    """
    Load FAISS index and chunks using plain faiss.read_index() + pickle.load().
    No LangChain dependency. Supports index.faiss or vector_index.faiss.
    """
    chunks_path = RAG_MODEL_DIR / "chunks.pkl"

    # Support either index.faiss or vector_index.faiss, whichever is present.
    candidate_index_paths = [
        RAG_MODEL_DIR / "index.faiss",
        RAG_MODEL_DIR / "vector_index.faiss",
    ]
    index_path = next((p for p in candidate_index_paths if p.exists()), None)

    if index_path is None or not chunks_path.exists():
        return None
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        index = faiss.read_index(str(index_path))

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        return {"index": index, "chunks": chunks, "embedder": embedder}
    except Exception as e:
        st.session_state["_rag_load_error"] = str(e)
        return None


# ----------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <h2 style="color:white; margin-bottom:0;">🧠 AI Suite</h2>
            <p style="color:#c7cde0; font-size:0.8rem;">Customer Intelligence Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "📈 Customer Segmentation",
            "🛒 Purchase Prediction (LSTM)",
            "😊 Sentiment Analysis (BERT)",
            "🤖 AI Customer Insights (RAG)",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.78rem; color:#c7cde0; line-height:1.6;">
        <b>Modules</b><br>
        • K-Means Segmentation<br>
        • LSTM Purchase Forecasting<br>
        • BERT Sentiment Engine<br>
        • FAISS RAG Assistant
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    render_header(
        "Customer Segmentation & Churn Prediction using AI",
        "An end-to-end AI platform combining clustering, deep learning, NLP, and retrieval-augmented generation "
        "to understand, predict, and engage customers intelligently.",
    )

    df = load_segments_csv()

    st.markdown('<div class="section-title">📊 Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    if df is not None:
        total_customers = len(df)
        n_segments = df["Segment"].nunique() if "Segment" in df.columns else (
            df["segment"].nunique() if "segment" in df.columns else "N/A"
        )
        churn_col = None
        for cand in ["Churn", "churn", "Churned", "is_churn"]:
            if cand in df.columns:
                churn_col = cand
                break
        if churn_col:
            try:
                churn_rate = f"{df[churn_col].astype(float).mean() * 100:.1f}%"
            except Exception:
                churn_rate = f"{(df[churn_col].astype(str).str.lower().isin(['1','yes','true'])).mean()*100:.1f}%"
        else:
            churn_rate = "N/A"
    else:
        total_customers, n_segments, churn_rate = "N/A", "N/A", "N/A"

    with c1:
        kpi_card("Total Customers", total_customers, "Loaded from dataset")
    with c2:
        kpi_card("Customer Segments", n_segments, "K-Means clusters")
    with c3:
        kpi_card("Churn Rate", churn_rate, "Estimated from data")
    with c4:
        kpi_card("AI Models", "4", "KMeans · LSTM · BERT · RAG")

    st.markdown('<div class="section-title">🧩 Project Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
        <p>
        This platform brings together four AI pipelines into a single business-ready dashboard:
        unsupervised <b>customer segmentation</b>, deep-learning based <b>purchase behavior forecasting</b>,
        transformer-based <b>sentiment analysis</b> of customer reviews, and a <b>retrieval-augmented
        generation (RAG)</b> assistant that answers business questions using your own data.
        The goal is to help teams identify high-value customers, anticipate churn, understand customer
        sentiment, and get instant AI-driven insights — all from one interface.
        </p>
        <span class="badge">K-Means Clustering</span>
        <span class="badge">LSTM (TensorFlow/Keras)</span>
        <span class="badge">BERT (Transformers)</span>
        <span class="badge">FAISS + RAG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">⚙️ AI Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Data Collection", "Ingest customer transactions, reviews, and profiles."),
        ("2", "Segmentation", "Cluster customers with K-Means & PCA on RFM features."),
        ("3", "Prediction", "Forecast purchase behavior using an LSTM sequence model."),
        ("4", "NLP & RAG", "Score sentiment with BERT and answer queries via FAISS RAG."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="workflow-step">
                    <div class="step-num">{num}</div>
                    <h4 style="margin:0.3rem 0; color:#182848;">{title}</h4>
                    <p style="font-size:0.82rem; color:#666;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">🚀 Explore the Modules</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        info_card(
            "📈 Customer Segmentation",
            "Visualize customer clusters, churn rate by segment, and explore the raw "
            "segmented dataset interactively.",
        )
        info_card(
            "😊 Sentiment Analysis (BERT)",
            "Type any customer review and get a real-time Positive / Neutral / Negative "
            "prediction with confidence scores.",
        )
    with m2:
        info_card(
            "🛒 Purchase Prediction (LSTM)",
            "Understand how a sequence model forecasts future purchase behavior from "
            "historical patterns.",
        )
        info_card(
            "🤖 AI Customer Insights (RAG)",
            "Ask natural-language business questions and get answers grounded in your "
            "own customer data via FAISS retrieval.",
        )

    render_footer()

# ============================================================================
# PAGE: CUSTOMER SEGMENTATION
# ============================================================================
elif page == "📈 Customer Segmentation":
    render_header(
        "📈 Customer Segmentation",
        "Unsupervised clustering to group customers by behavior, value, and churn risk.",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Segment Distribution")
        show_image_if_exists(IMG_SEGMENT_DIST, "Customer Segment Distribution")
    with col2:
        st.markdown("#### Churn Rate by Segment")
        show_image_if_exists(IMG_CHURN_BY_SEGMENT, "Churn Rate by Segment")

    with st.expander("📉 View Elbow Curve & Feature Importance"):
        e1, e2 = st.columns(2)
        with e1:
            show_image_if_exists(IMG_ELBOW, "Elbow Curve (Optimal K)")
        with e2:
            show_image_if_exists(IMG_FEATURE_IMPORTANCE, "Feature Importance")

    st.markdown('<div class="section-title">🗂️ Segmented Customer Data</div>', unsafe_allow_html=True)
    df = load_segments_csv()
    if df is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Rows", len(df))
        with c2:
            kpi_card("Columns", len(df.columns))
        with c3:
            seg_col = "Segment" if "Segment" in df.columns else ("segment" if "segment" in df.columns else None)
            kpi_card("Unique Segments", df[seg_col].nunique() if seg_col else "N/A")

        st.dataframe(df, use_container_width=True, height=420)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download customer_segments.csv",
            data=csv_bytes,
            file_name="customer_segments.csv",
            mime="text/csv",
        )
    else:
        st.warning("`customer_segments.csv` not found in the project root.")

    render_footer()

# ============================================================================
# PAGE: PURCHASE PREDICTION (LSTM)
# ============================================================================
elif page == "🛒 Purchase Prediction (LSTM)":
    render_header(
        "🛒 Purchase Prediction (LSTM)",
        "A recurrent deep learning model that forecasts future purchase behavior from sequential customer data.",
    )

    st.markdown('<div class="section-title">🧠 How the LSTM Model Works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
        <p>
        A Long Short-Term Memory (LSTM) network is a type of recurrent neural network designed to learn
        patterns in sequential data. Here, it is trained on each customer's historical purchase sequence
        (e.g. order frequency, spend, recency) to predict the likelihood and timing of their next purchase.
        LSTMs use gated memory cells to retain long-term dependencies while filtering out noise, making them
        well-suited for time-series customer behavior modeling.
        </p>
        <span class="badge">Sequential Modeling</span>
        <span class="badge">TensorFlow / Keras</span>
        <span class="badge">Time-Series Forecasting</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("#### Training Loss Curve")
        show_image_if_exists(IMG_LSTM_LOSS, "LSTM Training / Validation Loss")
    with col2:
        st.markdown("#### Model Details")
        model = load_lstm_model()
        if model is not None:
            kpi_card("Total Parameters", f"{model.count_params():,}")
            kpi_card("Number of Layers", len(model.layers))
            try:
                input_shape = model.input_shape
                kpi_card("Input Shape", str(input_shape))
            except Exception:
                pass

            with st.expander("🔍 View Full Model Summary"):
                buf = io.StringIO()
                model.summary(print_fn=lambda x: buf.write(x + "\n"))
                st.code(buf.getvalue(), language="text")
        else:
            st.warning(
                f"Model file not found at `models/lstm_purchase_model.keras`. "
                f"Place the trained model there to enable this section."
            )

    render_footer()

# ============================================================================
# PAGE: SENTIMENT ANALYSIS (BERT)
# ============================================================================
elif page == "😊 Sentiment Analysis (BERT)":
    render_header(
        "😊 Sentiment Analysis (BERT)",
        "Transformer-based sentiment classification for customer reviews and feedback.",
    )

    st.markdown('<div class="section-title">✍️ Try It Yourself</div>', unsafe_allow_html=True)
    review_text = st.text_area(
        "Enter a customer review",
        placeholder="e.g. 'The product quality was amazing but delivery took too long.'",
        height=120,
    )
    predict_clicked = st.button("🔮 Predict Sentiment", use_container_width=False)

    label_map_default = {0: "Negative", 1: "Neutral", 2: "Positive"}

    if predict_clicked:
        if not review_text.strip():
            st.warning("Please enter a review to analyze.")
        else:
            tokenizer, model, torch = load_bert_model()
            if model is None:
                st.error(
                    "BERT model not found at `models/bert_final`. "
                    "Please ensure the trained model directory is present."
                )
            else:
                with st.spinner("Analyzing sentiment..."):
                    inputs = tokenizer(
                        review_text, return_tensors="pt", truncation=True, padding=True, max_length=256
                    )
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()

                if isinstance(probs, float):
                    probs = [probs]

                id2label = getattr(model.config, "id2label", None)
                if id2label and len(id2label) == len(probs):
                    labels = [id2label[i].capitalize() for i in range(len(probs))]
                else:
                    labels = [label_map_default.get(i, f"Class {i}") for i in range(len(probs))]

                best_idx = int(np.argmax(probs))
                best_label = labels[best_idx]
                confidence = probs[best_idx] * 100

                css_class = "result-neutral"
                emoji = "😐"
                if "pos" in best_label.lower():
                    css_class, emoji = "result-positive", "😊"
                elif "neg" in best_label.lower():
                    css_class, emoji = "result-negative", "😠"

                st.markdown(
                    f"""
                    <div class="result-box {css_class}">
                        {emoji} Predicted Sentiment: {best_label} &nbsp;|&nbsp; Confidence: {confidence:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Class Probabilities")
                prob_df = pd.DataFrame({"Sentiment": labels, "Probability": probs}).sort_values(
                    "Probability", ascending=False
                )
                st.bar_chart(prob_df.set_index("Sentiment"))

    st.markdown('<div class="section-title">📊 Model Evaluation</div>', unsafe_allow_html=True)
    show_image_if_exists(IMG_BERT_CM, "BERT Confusion Matrix")

    render_footer()

# ============================================================================
# PAGE: AI CUSTOMER INSIGHTS (RAG)
# ============================================================================
elif page == "🤖 AI Customer Insights (RAG)":
    render_header(
        "🤖 AI Customer Insights (RAG)",
        "Ask natural-language questions and get answers grounded in your customer data via retrieval-augmented generation.",
    )

    st.markdown('<div class="section-title">💬 Ask a Question</div>', unsafe_allow_html=True)
    user_question = st.text_input(
        "What would you like to know about your customers?",
        placeholder="e.g. 'Why are high-value customers churning?'",
    )
    ask_clicked = st.button("🔍 Get Insights")

    if ask_clicked:
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            vectordb = load_rag_index()
            if vectordb is None:
                err = st.session_state.get("_rag_load_error")
                st.error(
                    "Could not load the FAISS knowledge base from `models/rag`. "
                    + (f"Details: {err}" if err else "Please ensure index.faiss or vector_index.faiss and chunks.pkl are present.")
                )
            else:
                with st.spinner("Retrieving relevant context..."):
                    try:
                        index = vectordb["index"]
                        chunks = vectordb["chunks"]
                        embedder = vectordb["embedder"]

                        query_vector = embedder.encode([user_question]).astype("float32")
                        distances, indices = index.search(query_vector, 4)

                        results = [
                            chunks[idx] for idx in indices[0]
                            if 0 <= idx < len(chunks)
                        ]
                    except Exception as e:
                        results = []
                        st.error(f"Retrieval failed: {e}")

                if results:
                    st.markdown('<div class="section-title">📚 Retrieved Context</div>', unsafe_allow_html=True)
                    combined_text = []
                    for i, doc in enumerate(results, start=1):
                        content = doc if isinstance(doc, str) else doc.get("text", str(doc))
                        combined_text.append(content)
                        st.markdown(
                            f'<div class="context-chunk"><b>Chunk {i}:</b> {content}</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown('<div class="section-title">🧾 AI Answer</div>', unsafe_allow_html=True)
                    summary = " ".join(combined_text)
                    snippet = summary[:600] + ("..." if len(summary) > 600 else "")
                    st.markdown(
                        f"""
                        <div class="info-card">
                        <p>Based on the most relevant records retrieved from the customer knowledge base:</p>
                        <p>{snippet}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown('<div class="section-title">💡 Business Recommendations</div>', unsafe_allow_html=True)
                    st.markdown(
                        """
                        <div class="info-card">
                        <p>
                        • Prioritize proactive outreach to customer segments showing recurring negative
                        sentiment or declining purchase frequency.<br>
                        • Personalize retention offers for high-value customers flagged as at-risk in the
                        segmentation and LSTM prediction modules.<br>
                        • Feed recurring themes from retrieved reviews into product and support teams to
                        close feedback loops quickly.<br>
                        • Monitor churn-prone segments weekly using the segmentation dashboard to catch
                        early warning signs.
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No relevant context could be retrieved for this question.")

    render_footer()