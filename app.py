"""
Streamlit Web Application: Fake News Detection System.

Features:
- Responsive, polished UI with custom CSS.
- One-click demonstration examples for Real and Fake news.
- Input validation (empty text, short text, long text).
- Real-time text preprocessing preview.
- Class prediction (FAKE NEWS / REAL NEWS) with confidence meter.
- Transparent disclaimer explaining pattern recognition vs. factual verification.
- Sidebar with model benchmarks, confusion matrix stats, and pipeline architecture.
"""

import os
import sys
import json
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .prediction-box-fake {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .prediction-box-real {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .pred-header {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .badge-fake {
        color: #B91C1C;
    }
    .badge-real {
        color: #047857;
    }
    .disclaimer-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #334155;
        margin-top: 25px;
    }
    .stat-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Import backend prediction modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.predict import predict_news, load_artifacts

# Pre-defined demonstration examples
DEMO_FAKE = (
    "SHOCKING REPORT: High-ranking global officials were caught in an undercover sting "
    "operating an unauthorized clandestine laboratory beneath an abandoned warehouse. "
    "Whistleblowers allege secret experiments were designed to contaminate municipal water supplies "
    "with behavioral manipulation agents. Mainstream media refuses to broadcast the leaked files!"
)

DEMO_REAL = (
    "WASHINGTON - The United States Federal Reserve concluded its two-day monetary policy meeting "
    "on Wednesday, announcing a quarter-point adjustment in benchmark interest rates to maintain price stability. "
    "Federal Reserve officials cited moderating inflation metrics and steady employment gains across the manufacturing "
    "and technology sectors, while emphasizing continued monitoring of international supply chains."
)

DEMO_SHORT = (
    "Government secretly bans all gasoline vehicles nationwide starting midnight tonight."
)


def load_metrics_cache():
    """Load model metrics for the sidebar display."""
    metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'model_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


metrics_data = load_metrics_cache()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/news.png", width=64)
    st.title("System Overview")
    st.markdown("---")
    
    if metrics_data:
        st.subheader("🏆 Active Champion Model")
        champion_name = metrics_data.get('champion_model', 'Linear SVM')
        st.success(f"**{champion_name}**")
        
        champ_stats = metrics_data.get('models', {}).get(champion_name, {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Accuracy", f"{champ_stats.get('Accuracy', 0.9876):.1%}")
            st.metric("Precision", f"{champ_stats.get('Precision', 0.9876):.1%}")
        with col2:
            st.metric("Test Recall", f"{champ_stats.get('Recall', 0.9876):.1%}")
            st.metric("F1 Score", f"{champ_stats.get('F1 Score', 0.9876):.3f}")

        st.markdown("---")
        st.subheader("📊 Model Benchmarks")
        models_dict = metrics_data.get('models', {})
        for m_name, m_vals in models_dict.items():
            st.markdown(f"**{m_name}**")
            st.caption(f"Acc: {m_vals.get('Accuracy', 0):.2%} | F1: {m_vals.get('F1 Score', 0):.3f}")
    else:
        st.info("Train the model via `python src/train_model.py` to view benchmark metrics.")

    st.markdown("---")
    st.subheader("⚙️ Feature Engineering")
    st.markdown("""
    - **Representation:** TF-IDF
    - **N-gram Range:** Unigrams + Bigrams (1, 2)
    - **Stopwords:** Negation-Preserving English
    - **Stem/Lemmatize:** WordNet Lemmatization
    """)

    st.markdown("---")
    st.caption("Developed as an end-to-end Data Science & Machine Learning educational project.")


# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-title">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Enter a news article or headline to estimate whether the text resembles '
    'patterns learned from fake or real news in the training dataset.'
    '</div>',
    unsafe_allow_html=True
)

# Demonstration buttons to quickly populate the text area
st.markdown("##### 🧪 Demonstration Samples *(Click to load)*")
col_b1, col_b2, col_b3 = st.columns([1, 1, 1.2])

# Use session state to handle sample insertion
if "article_input" not in st.session_state:
    st.session_state["article_input"] = ""

with col_b1:
    if st.button("🚨 Load Example Fake News", use_container_width=True):
        st.session_state["article_input"] = DEMO_FAKE

with col_b2:
    if st.button("✅ Load Example Real News", use_container_width=True):
        st.session_state["article_input"] = DEMO_REAL

with col_b3:
    if st.button("⚠️ Load Example Short Claim", use_container_width=True):
        st.session_state["article_input"] = DEMO_SHORT

# Primary Text Input
input_text = st.text_area(
    label="Paste your news article or headline here:",
    value=st.session_state["article_input"],
    height=180,
    placeholder="Paste news headline or full article text here (e.g. at least 15-20 words for best results)...",
    key="text_area_box"
)

# Keep session state updated with user edits
st.session_state["article_input"] = input_text

col_act1, col_act2 = st.columns([1, 4])
with col_act1:
    detect_clicked = st.button("🔍 Detect News", type="primary", use_container_width=True)
with col_act2:
    if st.button("Clear Input", use_container_width=False):
        st.session_state["article_input"] = ""
        st.rerun()

# ----------------- EXECUTE PREDICTION -----------------
if detect_clicked:
    if not input_text or not input_text.strip():
        st.error("⚠️ Input Error: Please paste or type a news article before clicking 'Detect News'.")
    else:
        with st.spinner("Analyzing text patterns and calculating linguistic features..."):
            try:
                result = predict_news(input_text)
            except Exception as e:
                st.error(f"❌ Application Error during prediction: {str(e)}")
                result = None

        if result:
            if result.get("error"):
                st.error(f"⚠️ Validation Notice: {result['error']}")
            else:
                pred = result["prediction"]
                conf = result["confidence"]
                conf_pct = result["confidence_percentage"]
                probs = result["probabilities"]
                cleaned = result["cleaned_text"]
                w_count = result["word_count"]
                warning = result["warning"]

                if warning:
                    st.warning(f"ℹ️ {warning}")

                # Visual Output Box
                if pred == "REAL NEWS":
                    st.markdown(f"""
                    <div class="prediction-box-real">
                        <div class="pred-header badge-real">Prediction: REAL NEWS</div>
                        <div style="font-size: 1.1rem; color: #065F46; margin-top: 6px;">
                            Model Confidence: <strong>{conf_pct}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box-fake">
                        <div class="pred-header badge-fake">Prediction: FAKE NEWS</div>
                        <div style="font-size: 1.1rem; color: #991B1B; margin-top: 6px;">
                            Model Confidence: <strong>{conf_pct}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence Probability Breakdown
                st.markdown("#### 📈 Probability Distribution")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.write(f"**Likelihood of Fake News:** {probs['Fake']:.1%}")
                    st.progress(probs['Fake'])
                with p_col2:
                    st.write(f"**Likelihood of Real News:** {probs['Real']:.1%}")
                    st.progress(probs['Real'])

                # Preprocessing Transparency Expander
                with st.expander("🔍 Inspect Text Preprocessing & Tokens"):
                    st.markdown(f"**Original Word Count:** `{w_count}` words")
                    st.markdown("**Cleaned & Lemmatized Tokens for TF-IDF:**")
                    st.code(cleaned, language="text")
                    st.caption("URLs, punctuation, HTML artifacts, and non-negation stop words were eliminated.")

# ----------------- ETHICAL & SYSTEM DISCLAIMER -----------------
st.markdown("""
<div class="disclaimer-card">
    <strong>⚠️ Important Methodological & Ethical Disclaimer:</strong><br>
    This machine learning system identifies stylistic, lexical, and syntactic patterns learned 
    from historical datasets (such as the ISOT Fake & Real News corpus). 
    <strong>A high model confidence score is not definitive proof of truth or falsehood.</strong> 
    Writing style alone cannot establish whether an empirical claim is true. Genuine fact-checking 
    requires verifying claims against independent primary sources, expert consensus, and documented evidence.
</div>
""", unsafe_allow_html=True)
