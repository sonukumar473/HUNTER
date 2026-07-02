import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="HUNTER - Fake Job Detector", layout="wide")

# ---------------------------
# 1. LOAD OR TRAIN MODEL (CACHED)
# ---------------------------
MODEL_PATH = "hunter_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    
    # Train from scratch
    df = pd.read_csv("fake_job_postings.csv")
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df["raw_text"] = df["description"].fillna("") + " " + df["requirements"].fillna("") + " " + df["company_profile"].fillna("")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    
    df["text_length"] = df["raw_text"].str.len()
    df["exclamation_count"] = df["raw_text"].str.count(r'!')
    df["dollar_count"] = df["raw_text"].str.count(r'\$')
    df["question_count"] = df["raw_text"].str.count(r'\?')
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df["clean_text"])
    X_extra = csr_matrix(df[["text_length", "exclamation_count", "dollar_count", "question_count"]].values)
    X = hstack([X_text, X_extra])
    y = df["fraudulent"]
    
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=0.5)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    return model, vectorizer

model, vectorizer = load_or_train_model()

# ---------------------------
# 2. SIMPLE CSS (removed heavy Three.js)
# ---------------------------
st.markdown("""
<style>
    .stApp { background: #0a0a0a; }
    .main .block-container {
        background: rgba(10,10,10,0.8);
        border: 1px solid rgba(0,255,65,0.2);
        border-radius: 12px;
        padding: 30px;
        backdrop-filter: blur(5px);
    }
    h1, h2, h3 { color: #00ff41; }
    textarea {
        background: #0d0d0d;
        border: 2px solid #00ff41;
        color: #00ff41;
        border-radius: 8px;
        padding: 12px;
    }
    .stButton > button {
        background: #00ff41;
        color: #0a0a0a;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 30px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(0,255,65,0.4);
        transform: scale(1.02);
    }
    .stSuccess, .stError, .stWarning {
        border-radius: 8px;
        border-left: 5px solid;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 3. HEADER
# ---------------------------
st.markdown("## 🔍 HUNTER — Fake Job Detector")
st.markdown("**ML-powered analysis** • Real‑time scam hunting")

# ---------------------------
# 4. PREDICTION FUNCTION
# ---------------------------
def clean_text_single(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_job(text):
    cleaned = clean_text_single(text)
    text_vec = vectorizer.transform([cleaned])
    extra = csr_matrix([[len(text), text.count('!'), text.count('$'), text.count('?')]])
    combined = hstack([text_vec, extra])
    prob = model.predict_proba(combined)[0]
    fake_prob = prob[1] * 100
    real_prob = prob[0] * 100
    if fake_prob > 35:
        return "⚠️ FAKE JOB HUNTED", fake_prob, "error"
    else:
        return "✅ REAL JOB", real_prob, "success"

# ---------------------------
# 5. INPUT AND ANALYSIS
# ---------------------------
user_input = st.text_area("Paste job description below:", height=180, placeholder="Minimum 20 characters...")

if st.button("⚡ HUNT SCAM", type="primary"):
    if len(user_input.strip()) < 20:
        st.warning("⚠️ Please paste at least 20 characters.")
    else:
        label, confidence, msg_type = predict_job(user_input)
        if msg_type == "error":
            st.error(f"### {label}\nConfidence: {confidence:.1f}%")
            # Show red flags
            red_flags = []
            if "urgent" in user_input.lower() or "immediately" in user_input.lower():
                red_flags.append("High‑urgency language (urgent, immediately)")
            if "$" in user_input:
                red_flags.append("Money symbols ($) – possible payment request")
            if "wire" in user_input.lower() or "transfer" in user_input.lower():
                red_flags.append("Wire transfer mentioned – major red flag")
            if "fee" in user_input.lower() or "payment" in user_input.lower():
                red_flags.append("Fee/payment request – never pay to apply")
            if red_flags:
                st.warning("🚨 **Red flags detected:**\n" + "\n".join(f"- {f}" for f in red_flags))
            else:
                st.info("🔍 No obvious keywords, but AI flagged this as suspicious.")
        else:
            st.success(f"### {label}\nConfidence: {confidence:.1f}%")
            st.info("✅ No major scam indicators detected. This posting appears legitimate.")

st.caption("HUNTER v3.0 • Always verify independently")