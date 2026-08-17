import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Sentiment Sense AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
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
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 0.5px;
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
    </style>
""", unsafe_allow_html=True)

# Load the pickle files
@st.cache_resource
def load_assets():
    model_path = "sentiment_model.pkl"
    vectorizer_path = "sentiment_vector.pkl"
    
    # Fallback to look for similar filenames if renamed
    if not os.path.exists(model_path):
        for f in os.listdir("."):
            if "model" in f.lower() and f.endswith(".pkl"):
                model_path = f
                break

    if not os.path.exists(vectorizer_path):
        for f in os.listdir("."):
            if ("vector" in f.lower() or "tfidf" in f.lower()) and f.endswith(".pkl"):
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

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3073/3073445.png", width=70)
    st.title("Sentiment AI")
    st.markdown("Analyze feedback, customer reviews, and general text sentiment instantly.")
    
    st.markdown("---")
    st.subheader("⚙️ Model Pipeline Status")
    if model and vectorizer:
        st.success("Artifacts Loaded (`sentiment_model.pkl` & `sentiment_vector.pkl`)")
    else:
        st.error("Model assets not detected")
        if err_msg:
            st.caption(f"Error: {err_msg}")
        st.info("Ensure `sentiment_model.pkl` and `sentiment_vector.pkl` are committed to your repository.")

    st.markdown("---")
    st.subheader("💡 Example Presets")
    sample_text = st.selectbox(
        "Try an example:",
        [
            "Select an example...",
            "The movie was absolutely fantastic, brilliant acting and storytelling!",
            "Completely terrible waste of time. Poor direction and horrible script.",
            "I really enjoyed the cinematography and the pacing was perfect."
        ]
    )

# Main UI
st.title("💬 Real-Time Sentiment Analysis")
st.caption("Multinomial Naive Bayes classifier trained with TF-IDF vectorization.")
st.markdown("---")

tab1, tab2 = st.tabs(["📝 Single Text Analysis", "📂 Batch CSV Analysis"])

with tab1:
    col_input, col_result = st.columns([1.2, 1], gap="large")

    with col_input:
        st.subheader("Input Text")
        default_val = sample_text if sample_text != "Select an example..." else ""
        user_input = st.text_area(
            "Enter text or feedback:",
            value=default_val,
            height=180,
            placeholder="Type your review or text here..."
        )

        analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True, type="primary")

    with col_result:
        st.subheader("Prediction Result")
        
        if analyze_btn:
            if not user_input.strip():
                st.warning("Please enter text to analyze.")
            elif not model or not vectorizer:
                st.error("Model or vectorizer file is missing from repository root.")
            else:
                with st.spinner("Classifying..."):
                    vec_text = vectorizer.transform([user_input])
                    prediction = model.predict(vec_text)[0]
                    
                    # Calculate probabilities
                    confidence = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vec_text)[0]
                        confidence = np.max(proba) * 100

                    pred_str = str(prediction).lower()
                    if pred_str in ["1", "positive", "pos"]:
                        label = "Positive"
                        css_class = "positive"
                        icon = "🟢 😊"
                    else:
                        label = "Negative"
                        css_class = "negative"
                        icon = "🔴 😞"

                    st.markdown(
                        f'<div class="sentiment-box {css_class}">{icon} {label.upper()}</div>',
                        unsafe_allow_html=True
                    )

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
            st.info("Enter text and click **Analyze Sentiment** to see the result.")

with tab2:
    st.subheader("Upload CSV for Bulk Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file containing a text column", type=["csv"])
    
    if uploaded_file and model and vectorizer:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head(3))
        
        text_column = st.selectbox("Select text column to analyze:", df.columns)
        
        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Processing batch records..."):
                texts = df[text_column].fillna("").astype(str)
                vec_batch = vectorizer.transform(texts)
                df["Predicted_Sentiment"] = model.predict(vec_batch)
                
                st.success("Batch processing complete!")
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write("**Sentiment Distribution**")
                    st.bar_chart(df["Predicted_Sentiment"].value_counts())
                with c2:
                    st.write("**Predictions Table**")
                    st.dataframe(df, use_container_width=True)
