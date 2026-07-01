import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from scipy.sparse import hstack, csr_matrix

print("="*70)
print("🔥 ULTIMATE FAKE JOB DETECTOR + PREDICTION SYSTEM")
print("="*70)

# ---------------------------
# 1. LOAD DATA (AUTOMATIC INSERT)
# ---------------------------
try:
    df = pd.read_csv("fake_job_postings.csv")
    print("✅ Dataset loaded successfully from the same folder!")
except FileNotFoundError:
    print("❌ ERROR: 'fake_job_postings.csv' not found in this folder.")
    print("📁 Make sure the CSV file is in the same folder as detector.py")
    exit()

# ---------------------------
# 2. ADVANCED TEXT CLEANING
# ---------------------------
print("🧹 Cleaning text (removing URLs, emails, numbers, symbols)...")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["raw_text"] = df["description"].fillna("") + " " + df["requirements"].fillna("") + " " + df["company_profile"].fillna("")
df["clean_text"] = df["raw_text"].apply(clean_text)

# ---------------------------
# 3. ENGINEERED FEATURES
# ---------------------------
print("📐 Adding extra features (length, exclamation marks, dollar signs)...")
df["text_length"] = df["raw_text"].str.len()
df["exclamation_count"] = df["raw_text"].str.count(r'!')
df["dollar_count"] = df["raw_text"].str.count(r'\$')
df["caps_count"] = df["raw_text"].str.count(r'[A-Z]')

# ---------------------------
# 4. TEXT TO NUMBERS (TF-IDF)
# ---------------------------
print("🔄 Converting text to numbers (15,000 features + phrases)...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 3),
)

X_text = vectorizer.fit_transform(df["clean_text"])
X_extra = csr_matrix(df[["text_length", "exclamation_count", "dollar_count", "caps_count"]].values)
X_combined = hstack([X_text, X_extra])
y = df["fraudulent"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# ---------------------------
# 5. HYPERPARAMETER TUNING (Logistic Regression)
# ---------------------------
print("⚙️ Tuning Logistic Regression...")
lr_tuned = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
grid_search = GridSearchCV(lr_tuned, {'C': [0.01, 0.1, 1, 10]}, scoring='recall', cv=3)
grid_search.fit(X_train, y_train)
best_lr = grid_search.best_estimator_
print(f"✅ Best Logistic C value: {grid_search.best_params_['C']}")

# ---------------------------
# 6. TRAIN OTHER MODELS
# ---------------------------
print("🧠 Training Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train, y_train)

print("🌲 Training Random Forest (wait 30-60 sec)...")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ---------------------------
# 7. EVALUATE & COMPARE
# ---------------------------
pred_lr = best_lr.predict(X_test)
pred_nb = nb.predict(X_test)
pred_rf = rf.predict(X_test)

print("\n" + "="*70)
print("🏆 3-WAY MODEL COMPARISON (Focus on catching Fakes)")
print("="*70)

models_dict = {"Logistic (Tuned)": pred_lr, "Naive Bayes": pred_nb, "Random Forest": pred_rf}
print(f"{'Model':<20} {'Accuracy':<12} {'Fake Recall':<15} {'Fake F1':<15}")
print("-"*70)

for name, pred in models_dict.items():
    print(f"{name:<20} {accuracy_score(y_test, pred):.2%}       {recall_score(y_test, pred):.2%}        {f1_score(y_test, pred):.2%}")

# ---------------------------
# 8. PICK THE BEST MODEL for our Prediction System
# ---------------------------
# We pick the one with the highest Recall (catches the most fakes)
recall_lr = recall_score(y_test, pred_lr)
recall_rf = recall_score(y_test, pred_rf)

if recall_rf >= recall_lr:
    final_model = rf
    final_model_name = "Random Forest"
else:
    final_model = best_lr
    final_model_name = "Logistic Regression (Tuned)"

print(f"\n✅ Best model for catching fakes: {final_model_name} (Recall: {max(recall_lr, recall_rf):.2%})")

# ---------------------------
# 9. THE PREDICTION SYSTEM (LIVE DEMO)
# ---------------------------
print("\n" + "="*70)
print("🛠️  PREDICTION SYSTEM ACTIVATED")
print("="*70)
print("You can now paste any job description and I'll tell you if it's FAKE or REAL.")
print("Type 'exit' to quit.")
print("="*70)

# This function takes a raw job text and returns a prediction
def predict_job(text):
    # Step 1: Clean the text
    cleaned = clean_text(text)
    
    # Step 2: Vectorize the text (using the same vectorizer we trained)
    text_vector = vectorizer.transform([cleaned])
    
    # Step 3: Calculate the extra features for this specific text
    text_len = len(text)
    exc_count = text.count('!')
    dol_count = text.count('$')
    cap_count = sum(1 for c in text if c.isupper())
    extra_vector = csr_matrix([[text_len, exc_count, dol_count, cap_count]])
    
    # Step 4: Combine both vectors
    combined = hstack([text_vector, extra_vector])
    
    # Step 5: Predict (0 = Real, 1 = Fake)
    prediction = final_model.predict(combined)[0]
    
    # Step 6: Get confidence (probability)
    probability = final_model.predict_proba(combined)[0]
    if prediction == 1:
        confidence = probability[1] * 100
    else:
        confidence = probability[0] * 100
    
    return prediction, confidence

# ---------------------------
# 10. LIVE INTERACTIVE LOOP
# ---------------------------
while True:
    print("\n" + "-"*50)
    user_input = input("📝 Paste job description (or 'exit'): ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Goodbye! Detection system closed.")
        break
    
    if len(user_input.strip()) < 20:
        print("⚠️ Please paste a longer job description (at least 20 characters).")
        continue
    
    # Predict
    pred, conf = predict_job(user_input)
    
    # Show result with a cool visual
    print("\n" + "="*50)
    if pred == 1:
        print(f"🚨 ⚠️  FAKE JOB DETECTED! (Confidence: {conf:.1f}%)")
        print("="*50)
        print("💡 Red flags: This posting looks suspicious based on language patterns.")
    else:
        print(f"✅ Legitimate Job (Confidence: {conf:.1f}%)")
        print("="*50)
        print("💡 This posting matches patterns of real, trusted job listings.")
    print("="*50)