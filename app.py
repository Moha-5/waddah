from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import json
import os
import pickle
import base64

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS


# -----------------------------------------
# Gmail OAuth config
# -----------------------------------------
GOOGLE_CLIENT_SECRETS_FILE = (
    "user_secret"
    "apps.googleusercontent.com.json"
)
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
REDIRECT_URI = ""

# -------------------------------------------------
# Flask setup
# -------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = "temporary_dev_key"  # change in production
CORS(app)

# -------------------------------------------------
# Load model & vectorizer once
# -------------------------------------------------
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


# -------------------------------------------------
# Core classification logic
# -------------------------------------------------
def classify_text(text: str):
    """
    Return:
      label: 'SPAM' / 'SUSPICIOUS' / 'LEGIT'
      score: spam probability in 0–100
    """
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    spam_prob = float(proba[1] * 100.0)  # class 1 = spam/phishing

    if spam_prob > 80:
        label = "SPAM"
    elif spam_prob >= 50:
        label = "SUSPICIOUS"
    else:
        label = "LEGIT"

    return label, spam_prob


def build_indicators(text: str, label: str, top_n: int = 8):
    """
    Rough explanation indicators from top words
    (like your CLI script).
    """
    if not hasattr(model, "feature_log_prob_"):
        return []

    # assume binary classification: 0=ham, 1=spam
    class_idx = 1 if label == "SPAM" else 0

    X = vectorizer.transform([text])
    X = X.tocoo()
    present_idx = X.col

    if len(present_idx) == 0:
        return []

    feature_names = vectorizer.get_feature_names_out()
    log_probs_present = model.feature_log_prob_[class_idx][present_idx]

    import numpy as np
    top_local_idx = np.argsort(log_probs_present)[-top_n:]
    top_feature_idx = present_idx[top_local_idx]

    words = [feature_names[i] for i in top_feature_idx]

    indicators = []
    for w in words:
        if label == "SPAM":
            indicators.append(f"Word '{w}' is strongly associated with phishing/spam emails.")
        elif label == "SUSPICIOUS":
            indicators.append(f"Word '{w}' contributes to the message being suspicious.")
        else:
            indicators.append(f"Word '{w}' is typical of legitimate messages.")

    return indicators


# -------------------------------------------------
# Gmail helpers  
# -------------------------------------------------
def get_user_credentials():
    """Load credentials from session; return Credentials or None."""
    token_json = session.get("google_token")
    if not token_json:
        return None
    data = json.loads(token_json)
    return Credentials.from_authorized_user_info(data, SCOPES)


def save_user_credentials(creds: Credentials):
    """Save credentials back to session (access + refresh tokens)."""
    session["google_token"] = creds.to_json()


def get_gmail_service():
    """Build a Gmail API client for this user."""
    creds = get_user_credentials()
    if not creds:
        return None

    if creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request

        creds.refresh(Request())
        save_user_credentials(creds)

    service = build("gmail", "v1", credentials=creds)
    return service


def extract_plain_text_from_message(msg):
    """
    Walk Gmail message payload and return combined text/plain body.
    """

    def walk_parts(parts):
        text = []
        for part in parts:
            mime_type = part.get("mimeType", "")
            body = part.get("body", {})
            data = body.get("data")

            if mime_type == "text/plain" and data:
                decoded = base64.urlsafe_b64decode(data.encode("UTF-8")).decode(
                    "UTF-8", errors="ignore"
                )
                text.append(decoded)
            elif part.get("parts"):
                text.append(walk_parts(part["parts"]))
        return "\n".join(filter(None, text))

    payload = msg.get("payload", {})
    if payload.get("mimeType", "").startswith("multipart/"):
        return walk_parts(payload.get("parts", []))
    else:
        body = payload.get("body", {}).get("data")
        if body:
            return base64.urlsafe_b64decode(body.encode("UTF-8")).decode(
                "UTF-8", errors="ignore"
            )
    return ""


# -------------------------------------------------
# Core routes
# -------------------------------------------------
@app.route("/")
def index():
    """Serve the main HTML frontend."""
    return send_from_directory("static", "Waddah-interface.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze-email", methods=["POST"])
def analyze_email():
    data = request.get_json(force=True) or {}

    sender = (data.get("sender_email") or "").strip()
    subject = (data.get("subject") or "").strip()
    content = (data.get("content") or "").strip()

    if not sender or not subject or not content:
        return jsonify({"error": "Missing fields"}), 400

    full_text = f"FROM: {sender}\nSUBJECT: {subject}\n\n{content}"

    label, score = classify_text(full_text)
    indicators = build_indicators(full_text, label)

    return jsonify(
        {
            "score": round(score, 2),
            "label": label,
            "indicators": indicators,
        }
    )


@app.route("/analyze-message", methods=["POST"])
def analyze_message():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Missing 'message' field"}), 400

    label, score = classify_text(message)
    indicators = build_indicators(message, label)

    return jsonify(
        {
            "score": round(score, 2),
            "label": label,
            "indicators": indicators,
        }
    )


# -------------------------------------------------
# Gmail routes  
# -------------------------------------------------
@app.route("/gmail/connect")
def gmail_connect():
    """Start OAuth flow and return Google auth URL."""
    flow = Flow.from_client_secrets_file(
        GOOGLE_CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    session["oauth_state"] = state
    return jsonify({"auth_url": authorization_url})


@app.route("/oauth2callback")
def oauth2callback():
    """Handle the OAuth callback from Google."""
    state = session.get("oauth_state")
    if not state:
        return "Missing OAuth state", 400

    flow = Flow.from_client_secrets_file(
        GOOGLE_CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=state,
        redirect_uri=REDIRECT_URI,
    )

    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)

    creds = flow.credentials
    save_user_credentials(creds) 

    # Redirect back to the frontend (main app)
    return send_from_directory("static", "Waddah-interface.html")


@app.route("/gmail/messages", methods=["GET"])
def gmail_list_messages():
    """List recent inbox messages for this user."""
    service = get_gmail_service()
    if not service:
        return jsonify({"error": "NOT_CONNECTED"}), 401

    query = request.args.get("q", "")
    max_results = int(request.args.get("maxResults", 20))

    result = (
        service.users()
        .messages()
        .list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=max_results,
            q=query or None,
        )
        .execute()
    )

    messages = result.get("messages", [])
    items = []

    for m in messages:
        msg = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=m["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )

        headers = msg.get("payload", {}).get("headers", [])
        hdr = {h["name"]: h["value"] for h in headers}

        items.append(
            {
                "id": msg["id"],
                "snippet": msg.get("snippet", ""),
                "from": hdr.get("From", ""),
                "subject": hdr.get("Subject", ""),
                "date": hdr.get("Date", ""),
            }
        )

    return jsonify({"messages": items})


@app.route("/gmail/messages/<message_id>/analyze", methods=["GET"])
def gmail_analyze_message(message_id):
    """Fetch a Gmail message by ID, analyze content with your model."""
    service = get_gmail_service()
    if not service:
        return jsonify({"error": "NOT_CONNECTED"}), 401

    msg = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=message_id,
            format="full",
        )
        .execute()
    )

    headers = msg.get("payload", {}).get("headers", [])
    hdr = {h["name"]: h["value"] for h in headers}
    from_addr = hdr.get("From", "")
    subject = hdr.get("Subject", "")

    body_text = extract_plain_text_from_message(msg)
    full_text = f"FROM: {from_addr}\nSUBJECT: {subject}\n\n{body_text}"

    label, score = classify_text(full_text)
    indicators = build_indicators(full_text, label)

    return jsonify(
        {
            "score": round(score, 2),
            "label": label,
            "indicators": indicators,
            "subject": subject,
            "from": from_addr,
        }
    )


# -------------------------------------------------
# Run server
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)

