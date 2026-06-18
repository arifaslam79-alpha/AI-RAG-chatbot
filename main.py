import os
import uuid
import json
import logging
from typing import List, Dict, Any
from flask import Flask, render_template, request, jsonify, make_response
from dotenv import load_dotenv
import openai

# Optional Redis
import redis

# Optional vector store (FAISS)
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

load_dotenv()

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or .env file")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Redis setup: if REDIS_URL present, try to connect; otherwise fallback to in-memory
REDIS_URL = os.getenv("REDIS_URL")
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", 60 * 60 * 24 * 30))  # default 30 days

redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        LOG.info("Connected to Redis at %s", REDIS_URL)
    except Exception as e:
        LOG.warning("Could not connect to Redis at %s: %s. Falling back to in-memory.", REDIS_URL, e)
        redis_client = None
else:
    LOG.info("REDIS_URL not set — using in-memory conversation store")

# In-memory session storage (for fallback/demo only).
CONVERSATIONS: Dict[str, List[Dict[str, str]]] = {}

SYSTEM_PROMPT = """You are a helpful, safe, and conservative health assistant. 
Give general health information, education, and guidance, but do NOT provide a medical diagnosis or definitive medical advice.
Always include a clear disclaimer that you are not a substitute for professional medical care and advise users to consult a licensed healthcare professional for specific concerns.
If the user describes emergency symptoms (chest pain, severe shortness of breath, severe bleeding, loss of consciousness), instruct them to seek emergency care immediately and do not attempt to triage further yourself.
Be concise, use plain language, and ask clarifying questions when necessary.
"""

EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath", "severe bleeding",
    "loss of consciousness", "unresponsive", "fainting", "not breathing", "no pulse"
]

def contains_emergency(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in EMERGENCY_KEYWORDS)

# Redis key helpers
def _redis_key(session_id: str) -> str:
    return f"hc:conv:{session_id}"

def _default_messages() -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]

# Storage helpers that abstract Redis vs in-memory
def get_session_messages(session_id: str) -> List[Dict[str, str]]:
    if redis_client:
        try:
            raw = redis_client.get(_redis_key(session_id))
            if raw:
                msgs = json.loads(raw)
                return msgs
            else:
                return _default_messages()
        except Exception as e:
            LOG.warning("Redis get error: %s — falling back to in-memory", e)
    return CONVERSATIONS.get(session_id, _default_messages())

def append_message(session_id: str, role: str, content: str) -> None:
    msgs = get_session_messages(session_id)
    if not msgs:
        msgs = _default_messages()
    msgs.append({"role": role, "content": content})
    if len(msgs) > 30:
        msgs = [msgs[0]] + msgs[-29:]
    if redis_client:
        try:
            redis_client.setex(_redis_key(session_id), REDIS_TTL_SECONDS, json.dumps(msgs))
            return
        except Exception as e:
            LOG.warning("Redis set error: %s — falling back to in-memory", e)
    CONVERSATIONS[session_id] = msgs

# --------------------------
# Vector store (FAISS) utils
# --------------------------
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

vector_index = None
vector_docs: List[Dict[str, Any]] = []

def load_vector_store(path: str):
    global vector_index, vector_docs
    if not FAISS_AVAILABLE:
        LOG.warning("FAISS not available; skipping vector store load.")
        return
    idx_path = os.path.join(path, "index.faiss")
    docs_path = os.path.join(path, "docs.json")
    if os.path.exists(idx_path) and os.path.exists(docs_path):
        try:
            vector_index = faiss.read_index(idx_path)
            with open(docs_path, "r", encoding="utf-8") as fh:
                vector_docs = json.load(fh)
            LOG.info("Loaded vector store from %s (n=%d)", path, len(vector_docs))
        except Exception as e:
            LOG.exception("Failed to load vector store: %s", e)
            vector_index = None
            vector_docs = []
    else:
        LOG.info("No vector store found at %s", path)

def embed_text(text: str) -> List[float]:
    resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
    return resp["data"][0]["embedding"]

def retrieve_context(query: str, k: int = RAG_TOP_K) -> List[Dict[str, str]]:
    """
    Returns top-k document chunks as a list of dicts: {source, chunk_index, text, score}
    If vector store not loaded, returns empty list.
    """
    if vector_index is None or not FAISS_AVAILABLE or not vector_docs:
        return []

    q_emb = np.array(embed_text(query), dtype="float32")
    # Normalize for cosine (we stored normalized vectors)
    faiss.normalize_L2(q_emb.reshape(1, -1))
    D, I = vector_index.search(q_emb.reshape(1, -1), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(vector_docs):
            continue
        doc = vector_docs[idx].copy()
        doc["score"] = float(score)
        results.append(doc)
    return results

# Load vector store on startup (best-effort)
if FAISS_AVAILABLE:
    load_vector_store(VECTOR_STORE_PATH)
else:
    LOG.info("FAISS library not installed; RAG disabled.")

# --------------------------
# Flask endpoints
# --------------------------
@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    return resp

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "message required"}), 400

    user_message = data["message"].strip()
    session_id = data.get("session_id") or request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    # Emergency guard
    if contains_emergency(user_message):
        reply = (
            "I may be limited in what I can do, but it sounds like you might be experiencing a medical emergency. "
            "Please call your local emergency number (e.g., 911) or go to the nearest emergency department right now. "
            "If possible, contact someone nearby for help."
        )
        append_message(session_id, "user", user_message)
        append_message(session_id, "assistant", reply)
        return jsonify({"reply": reply, "session_id": session_id})

    append_message(session_id, "user", user_message)
    messages = get_session_messages(session_id)

    # RAG: retrieve supporting context and inject into the prompt
    retrieved = retrieve_context(user_message)
    if retrieved:
        # Build a short context string with sources
        ctx_parts = []
        for i, r in enumerate(retrieved, start=1):
            src = r.get("source", "unknown")
            idx = r.get("chunk_index", 0)
            txt = r.get("text", "")
            score = r.get("score", 0.0)
            ctx_parts.append(f"[{i}] source: {src} (chunk {idx}, score={score:.4f})\n{txt}\n")
        context_block = "\n\n---\n\n".join(ctx_parts)

        # Add an assistant/system message that provides the context and instructs how to use it.
        # We append a system-like message so the model knows these are retrieved facts to use.
        rag_message = {
            "role": "system",
            "content": (
                "Use the following retrieved documents from the knowledge base to help answer the user. "
                "Only use information from these documents when they are directly relevant, and cite the source(s) by file name when you do. "
                "If the documents are not relevant, you may ignore them. Do not hallucinate additional sources.\n\n"
                f"RETRIEVED DOCUMENTS:\n\n{context_block}"
            ),
        }
        # For safety, we prepend the RAG message after the system prompt but before the conversation.
        # To avoid duplicating the system prompt, create a new message list where the first element is original system.
        base_system = messages[0] if messages else {"role": "system", "content": SYSTEM_PROMPT}
        rest = messages[1:] if len(messages) > 1 else []
        augmented_messages = [base_system, rag_message] + rest
    else:
        augmented_messages = messages

    try:
        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=augmented_messages,
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "500")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        )
        assistant_text = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        LOG.exception("OpenAI request failed")
        assistant_text = (
            "Sorry, I couldn't reach the language model service. Please try again later."
        )

    append_message(session_id, "assistant", assistant_text)

    resp = jsonify({"reply": assistant_text, "session_id": session_id})
    resp.set_cookie("session_id", session_id, httponly=True, samesite="Lax")
    return resp

if __name__ == "__main__":
    # Local dev helper: show whether redis is configured
    if redis_client:
        LOG.info("Using Redis for session storage. TTL seconds=%s", REDIS_TTL_SECONDS)
    else:
        LOG.info("Using in-memory session storage (not persistent).")
    if FAISS_AVAILABLE and vector_index is not None:
        LOG.info("RAG enabled. Vector store path=%s", VECTOR_STORE_PATH)
    else:
        LOG.info("RAG disabled (FAISS or index not available).")
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
