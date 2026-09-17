"""
server.py — a thin HTTP wrapper around query.py, for the React UI.

Nothing about RAG changes here: this file doesn't reimplement retrieval
or generation, it just exposes the same retrieve() / build_prompt() /
generate_answer() functions from query.py over HTTP, so a browser-based
UI (which can't import Python modules directly) can call them.

The embedding model + FAISS index are loaded ONCE at startup (not per
request) — reloading an ~80MB model on every keystroke would make the
UI painfully slow. This is the same reasoning as query.py loading them
once per CLI invocation, just stretched across the server's lifetime
instead of a single run.

Run with:
    ./.venv/bin/uvicorn server:app --reload --port 8000
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import config
import query


app = FastAPI(title="HR RAG API")

# The React dev server (Vite) runs on a different port (5173) than this
# API (8000). Browsers block cross-origin requests by default (CORS) --
# this explicitly allows the frontend's origin to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
_index, _metadata = query.load_index_and_metadata()
print("Ready.")


class QueryRequest(BaseModel):
    query: str
    user_id: str = config.CURRENT_USER_ID
    provider: str = config.LLM_PROVIDER
    # Deliberately NO `role` field here. Role is looked up server-side
    # from config.USERS[user_id] in run_query(), never taken from the
    # request body -- if a client could just set {"role": "hr"} on any
    # request, the RBAC check would be trivially bypassable. This is the
    # same "never trust the client for identity" principle a real JWT
    # verification step would enforce.


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_file: str
    doc_type: str
    employee_id: Optional[str]
    distance: float
    text: str
    relevant: bool  # False = dropped by the employee-ID relevance guardrail, not used in the prompt


class QueryResponse(BaseModel):
    query: str
    user_id: str
    role: str  # server-resolved, trustworthy -- echoed back so the UI can display it
    provider: str
    retrieved_chunks: List[RetrievedChunk]
    guardrail_triggered: bool  # True = no chunk was relevant enough; answer was NOT LLM-generated
    answer: str


@app.get("/api/meta")
def get_meta():
    """
    Tells the frontend what options to offer in its dropdowns, so those
    choices live in config.py (one place) instead of being duplicated as
    hardcoded strings inside the React app. Each user's role is included
    here for display purposes only -- it is NOT what run_query() trusts;
    run_query() re-looks-up the role itself from config.USERS every time.
    """
    return {
        "users": [{"id": user_id, "role": info["role"]} for user_id, info in config.USERS.items()],
        "providers": ["anthropic", "ollama"],
        "default_user_id": config.CURRENT_USER_ID,
        "default_provider": config.LLM_PROVIDER,
        "top_k": config.TOP_K,
    }


@app.post("/api/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    """
    The same three-step pipeline query.py's main() runs on the CLI:
    retrieve -> build prompt -> generate answer. The only new thing here
    is that `user_id` and `provider` come from the HTTP request body
    (i.e. whatever the UI's dropdowns are set to) instead of config.py's
    hardcoded defaults.
    """
    user_info = config.USERS.get(request.user_id)
    if user_info is None:
        raise HTTPException(status_code=400, detail=f"Unknown user_id: {request.user_id!r}")
    role = user_info["role"]

    if query.is_pure_greeting(request.query):
        return QueryResponse(
            query=request.query,
            user_id=request.user_id,
            role=role,
            provider=request.provider,
            retrieved_chunks=[],
            guardrail_triggered=False,
            answer=query.GREETING_RESPONSE,
        )

    retrieved = query.retrieve(
        request.query, _model, _index, _metadata, config.TOP_K, request.user_id, role
    )
    relevant = query.apply_relevance_guardrail(request.query, retrieved)

    # Same guardrail as the CLI: if the question named a specific
    # employee ID and nothing retrieved actually belongs to that ID,
    # don't call the LLM at all -- see query.py's apply_relevance_guardrail
    # for why (this is what fixes the EMP99999 CTC misattribution bug).
    guardrail_triggered = len(relevant) == 0
    if guardrail_triggered:
        answer = query.NO_RELEVANT_CONTEXT_MESSAGE
    else:
        prompt = query.build_prompt(request.query, relevant)
        answer = query.generate_answer(prompt, provider=request.provider)

    relevant_ids = {chunk["chunk_id"] for chunk in relevant}
    annotated_chunks = [
        {**chunk, "relevant": chunk["chunk_id"] in relevant_ids} for chunk in retrieved
    ]

    return QueryResponse(
        query=request.query,
        user_id=request.user_id,
        role=role,
        provider=request.provider,
        retrieved_chunks=annotated_chunks,
        guardrail_triggered=guardrail_triggered,
        answer=answer,
    )
