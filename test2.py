import joblib
import os
import numpy as np
import re

# ---------- CONFIG ----------

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# 👇 Put your email text here
EMAIL_TEXT = """
Hi team,

Please find attached the meeting notes from yesterday.
Let me know if you have any questions.

Best regards,
Ahmed
"""

# ---------- LOAD MODEL & VECTORIZER ----------

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer

# ---------- CLASSIFICATION LOGIC ----------

def classify_email(text: str):
    model, vectorizer = load_artifacts()

    # Vectorize text
    X = vectorizer.transform([text])

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support probability prediction (predict_proba).")

    # Assumes binary model: [legit, spam]
    spam_probability = model.predict_proba(X)[0][1] * 100  # percentage

    # Apply thresholds
    if spam_probability > 80:
        label = "SPAM"
    elif spam_probability >= 50:
        label = "SUSPICIOUS"
    else:
        label = "LEGIT"

    return label, spam_probability

# ---------- IMPORTANT WORDS FOR THIS EMAIL ----------

def get_top_class_words_for_email(model, vectorizer, text: str, label: str, top_n: int = 10):
    # Works only for Naive Bayes-style models
    if not hasattr(model, "feature_log_prob_"):
        return []

    # Assume class index 0 = ham/legit, 1 = spam
    # If label is SPAM → use spam class; otherwise → use ham class
    class_idx = 1 if label == "SPAM" else 0

    X = vectorizer.transform([text])
    X = X.tocoo()  # sparse -> COO to get non-zero indices
    present_idx = X.col  # indices of words that appear in this email

    if len(present_idx) == 0:
        return []

    feature_names = vectorizer.get_feature_names_out()

    # log P(word | class) for the words that are present
    log_probs_present = model.feature_log_prob_[class_idx][present_idx]

    # pick top N among present words
    top_local_idx = np.argsort(log_probs_present)[-top_n:]
    top_feature_idx = present_idx[top_local_idx]

    return [feature_names[i] for i in top_feature_idx]

def highlight_words_in_text(text: str, words):
    highlighted = text
    # longer words first to avoid partial overlaps
    for w in sorted(set(words), key=len, reverse=True):
        pattern = rf"\b{re.escape(w)}\b"
        highlighted = re.sub(pattern, f"[{w.upper()}]", highlighted, flags=re.IGNORECASE)
    return highlighted

# ---------- MAIN ----------

if __name__ == "__main__":
    text = EMAIL_TEXT.strip()

    if not text:
        print("❌ EMAIL_TEXT is empty.")
    else:
        # classify
        label, score = classify_email(text)

        # load model/vectorizer for explanation
        model, vectorizer = load_artifacts()

        # get important words for THIS email & THIS label
        top_words = get_top_class_words_for_email(model, vectorizer, text, label, top_n=10)
        highlighted_text = highlight_words_in_text(text, top_words)

        print("Email content (highlighted):")
        print("-" * 40)
        print(highlighted_text)
        print("-" * 40)

        print(f"Classification : {label}")
        print(f"Spam probability: {score:.2f}%")

        if top_words:
            if label == "SPAM":
                print("\nWords pushing it towards SPAM:")
            else:
                print("\nWords pushing it towards LEGIT:")
            print(", ".join(top_words))
