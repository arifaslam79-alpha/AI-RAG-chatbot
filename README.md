# Health Chatbot (Python + Flask) — Demo

This is a minimal demo health chatbot implemented in Python with Flask and OpenAI's Chat API.

Important
- This is for demonstration / educational purposes only. It is NOT a medical device and does not provide medical diagnoses.
- Do not use this in production for real patients without appropriate safety review, logging, testing, and privacy/compliance measures (e.g., HIPAA).

Requirements
- Python 3.9+
- An OpenAI API key (set as OPENAI_API_KEY)

Install
1. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # or venv\\Scripts\\activate on Windows

2. Install requirements:
   pip install -r requirements.txt

3. Create a `.env` file with:
   OPENAI_API_KEY=sk-...

Run
1. Start the server:
   python main.py

2. Open http://127.0.0.1:5000 in your browser.

How it works (high level)
- Frontend sends user messages to POST /chat.
- Server checks for emergency keywords; if present, it returns an emergency instruction and does not call the LLM.
- Otherwise, the server assembles conversation messages (keeps short in-memory history) and calls OpenAI ChatCompletion with a system prompt instructing safe behavior.
- Responses are returned to the UI and kept in-memory per session.

Next steps / improvements
- Replace in-memory storage with a database (Postgres/Redis) for persistence and scale.
- Add authentication and logging/auditing.
- Add a retrieval-augmented component (vector DB) feeding medical FAQ or vetted resources for accurate citations.
- Integrate a clinical triage flow and rules engine (and have clinicians review).
- Add rate limiting, monitoring, and safe-failover behavior.
- If handling PHI, ensure hosting, access controls, encryption, and a BAA with the model provider.

RAG (Retrieval-Augmented Generation) notes
- Use the `ingest_faqs.py` script to build a FAISS vector store from a folder of .md/.txt files.
- Place your documents in `./faqs` and run:
  - python ingest_faqs.py --source ./faqs --output ./vector_store
- Mount `./vector_store` into the container (or place it in the project root) so the app can load it on startup.
- The app will retrieve relevant chunks and include them in prompts; it instructs the model to cite source filenames when using retrieved text.

Docker
- A Dockerfile and docker-compose.yml are included. The compose file also runs Redis for session persistence.

Security & Privacy
- Conversations and documents are sent to OpenAI for embeddings and completion. If you need to avoid sending PHI, use local embeddings and LLMs.

