import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Sentiment Sense AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Modern UI with cards, clean typography, and smooth colors)
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 600;
    }
    .positive {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .negative {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
    .neutral {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer with caching
@st.cache_resource
def load_assets():
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    # Fallback checks if file names differ slightly
    if not os.path.exists(model_path):
        for f in os.listdir("."):
            if "model" in f.lower() and f.endswith(".pkl"):
                model_path = f
                break

    if not os.path.exists(vectorizer_path):
        for f in os.listdir("."):
            if ("vectorizer" in f.lower() or "tfidf" in f.lower() or "cv" in f.lower()) and f.endswith(".pkl"):
                vectorizer_path = f
                break

    try:
        with open(model_path, "rb") as f_model:
            model = pickle.load(f_model)
        with open(vectorizer_path, "rb") as f_vec:
            vectorizer = pickle.load(f_vec)
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

model, vectorizer, err_msg = load_assets()

# Sidebar Setup
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3073/3073445.png", width=70)
    st.title("Sentiment AI")
    st.markdown("Analyze text tone and emotions instantly using machine learning.")
    
    st.markdown("---")
    st.subheader("⚙️ System Status")
    if model and vectorizer:
        st.success("Artifacts Loaded Successfully")
    else:
        st.error("Missing `.pkl` files")
        if err_msg:
            st.caption(f"Error: {err_msg}")
        st.info("Ensure `model.pkl` and `vectorizer.pkl` exist in the root folder.")

    st.markdown("---")
    st.subheader("💡 Try Samples")
    sample_text = st.selectbox(
        "Select a quick sample:",
        [
            "Select an example...",
            "The customer service was phenomenal, absolutely loved it!",
            "Completely disappointed. It stopped working after two days.",
            "The delivery was on time. The package arrived safely."
        ]
    )

# Header Section
st.title("💬 Real-Time Sentiment Analysis")
st.caption("Extract emotional polarity, classification confidence, and key metrics from raw text.")

st.markdown("---")

tab1, tab2 = st.tabs(["📝 Single Text Analysis", "📂 Batch CSV Analysis"])

with tab1:
    col_input, col_result = st.columns([1.2, 1], gap="large")

    with col_input:
        st.subheader("Input Text")
        
        # Populate with sample or user input
        default_val = sample_text if sample_text != "Select an example..." else ""
        user_input = st.text_area(
            "Enter customer review, tweet, or feedback:",
            value=default_val,
            height=180,
            placeholder="Type or paste your text here..."
        )

        col_btn, col_clear = st.columns([1, 1])
        with col_btn:
            analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True, type="primary")

    with col_result:
        st.subheader("Prediction")
        
        if analyze_btn:
            if not user_input.strip():
                st.warning("Please enter some text to analyze.")
            elif not model or not vectorizer:
                st.error("Model assets not loaded. Please verify your `.pkl` files.")
            else:
                with st.spinner("Classifying sentiment..."):
                    # Vectorize input
                    vec_text = vectorizer.transform([user_input])
                    prediction = model.predict(vec_text)[0]
                    
                    # Probability calculation (if supported by model)
                    confidence = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vec_text)[0]
                        confidence = np.max(proba) * 100

                    # Format label
                    pred_str = str(prediction).lower()
                    if pred_str in ["1", "positive", "pos"]:
                        label = "Positive"
                        css_class = "positive"
                        icon = "🟢 😊"
                    elif pred_str in ["0", "negative", "neg", "-1"]:
                        label = "Negative"
                        css_class = "negative"
                        icon = "🔴 😞"
                    else:
                        label = "Neutral"
                        css_class = "neutral"
                        icon = "⚪ 😐"

                    # Display card
                    st.markdown(
                        f'<div class="sentiment-box {css_class}">{icon} {label}</div>',
                        unsafe_allow_html=True
                    )

                    # Display metrics
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.caption("Word Count")
                        st.subheader(len(user_input.split()))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with m_col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.caption("Confidence")
                        st.subheader(f"{confidence:.1f}%" if confidence is not None else "N/A")
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Enter text and click **Analyze Sentiment** to see the classification.")

with tab2:
    st.subheader("Upload CSV for Bulk Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file with a text column", type=["csv"])
    
    if uploaded_file and model and vectorizer:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head(3))
        
        text_column = st.selectbox("Select text column for prediction:", df.columns)
        
        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Processing batch data..."):
                texts = df[text_column].fillna("").astype(str)
                vec_batch = vectorizer.transform(texts)
                df["Predicted_Sentiment"] = model.predict(vec_batch)
                
                st.success("Analysis complete!")
                
                col_chart, col_table = st.columns([1, 2])
                with col_chart:
                    st.write("**Sentiment Distribution**")
                    st.bar_chart(df["Predicted_Sentiment"].value_counts())
                
                with col_table:
                    st.write("**Processed Records**")
                    st.dataframe(df, use_container_width=True)
