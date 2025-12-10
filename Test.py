import joblib
import os

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

# ---------- MAIN ----------

if __name__ == "__main__":
    text = EMAIL_TEXT.strip()

    if not text:
        print("❌ EMAIL_TEXT is empty.")
    else:
        label, score = classify_email(text)

        print("Email content:")
        print("-" * 40)
        print(text)
        print("-" * 40)

        print(f"Classification : {label}")
        print(f"Spam probability: {score:.2f}%")
