"""
Central place for constants used by ingest.py and query.py.

Keeping these in one file means you can tweak chunk size, top_k, etc.
and see how retrieval quality changes, without hunting through the code.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
# Path(__file__).parent gives the folder this config.py lives in, so the
# scripts work no matter what directory you run `python ingest.py` from.
PROJECT_DIR = Path(__file__).parent
HR_DOCS_DIR = PROJECT_DIR / "hr_docs"
FAISS_INDEX_PATH = PROJECT_DIR / "faiss_index.bin"
METADATA_PATH = PROJECT_DIR / "chunks_metadata.json"

# --- Chunking ------------------------------------------------------------
# Fixed-size chunking measured in characters (not tokens/words) because
# it's the simplest thing that works for a first pass — no tokenizer
# needed. 500 chars is roughly 100-125 words, small enough that a chunk
# stays topically focused, large enough to hold a full paragraph.
CHUNK_SIZE = 500

# Overlap between consecutive chunks. Without overlap, a sentence that
# straddles the boundary between chunk N and chunk N+1 gets split in
# half, and neither half alone has enough context to be useful when
# retrieved. Overlap duplicates a small window of text at each boundary
# so that a sentence which got cut in one chunk usually appears intact
# in a neighboring chunk too.
CHUNK_OVERLAP = 50

# --- Embedding model -----------------------------------------------------
# all-MiniLM-L6-v2: a small (~80MB) sentence-transformers model that runs
# fully locally (no API key, no network call at query time) and produces
# 384-dimensional embeddings. It's not as strong as large commercial
# embedding models, but it's free, fast, and plenty good for learning the
# mechanics of RAG on a handful of documents.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Retrieval -----------------------------------------------------------
# How many chunks to retrieve per query. Configurable here so you can
# experiment: too low and you might miss the right chunk, too high and
# you dilute the LLM's context with irrelevant text (and pay for more
# tokens).
TOP_K = 3

# --- Relevance guardrail (see query.py's apply_relevance_guardrail) ------
# We initially tried a blanket embedding-distance cutoff here to reject
# "no good match" retrievals before they reach the LLM. It had to be
# abandoned: with a corpus this small and a lightweight embedding model,
# genuinely relevant and genuinely irrelevant matches land in
# *overlapping* distance ranges -- e.g. a real answer to "What is my
# CTC?" scored a WORSE (higher) distance than a genuinely irrelevant
# match for an unrelated question. No single cutoff can separate them
# without either missing real bugs or blocking real answers. See
# query.py for the more targeted check that replaced it.
EMPLOYEE_ID_PATTERN = r"\bEMP\d+\b"

# --- Generation ------------------------------------------------------------
# Which LLM does the final "read the retrieved chunks, write an answer"
# step. Two options, same prompt either way — only the API call in
# query.py differs:
#   "anthropic" -> Claude API, hosted, needs ANTHROPIC_API_KEY in .env
#   "ollama"    -> a model running locally via Ollama, no API key, no
#                  network call, but only as good as the local model
#                  and hardware you run it on.
# Switch by changing this one value.
LLM_PROVIDER = "ollama"  # "anthropic" or "ollama" — no .env key yet, so defaulting to ollama

ANTHROPIC_MODEL = "claude-sonnet-5"

# Ollama serves an OpenAI-style local HTTP API (default port 11434) with
# no auth — that's what makes it possible to swap in with no API key.
# Matches what's already pulled on this machine (`ollama list`).
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Access control (Step 5 — naive simulation, not real auth) -----------
# In a real system, the current user's identity would come from an
# authenticated session (e.g. a verified JWT), and access rules would be
# enforced server-side per request. Here we just hardcode it so we can
# see the *filtering logic* work before building real AuthN/AuthZ later.
#
# USERS is an RBAC extension on top of Step 5: it adds a ROLE to each
# user, not just an id. An "employee" role is still restricted to their
# own personal documents only (Step 5's original behavior, unchanged).
# An "hr" role can see ANY employee's personal documents -- like a real
# HR admin would need to -- bypassing the ownership check entirely. This
# is exactly the "role" claim that sat unused in the JWT demo payload
# earlier ({"employee_id": ..., "role": ...}) -- now it's consumed.
#
# Same caveat as Step 5: USERS/CURRENT_USER_ID are a hardcoded stand-in
# for a real user/session database, not real authentication.
USERS = {
    "EMP10453": {"role": "employee"},
    "EMP99999": {"role": "employee"},
    "HR001": {"role": "hr"},
}
CURRENT_USER_ID = "EMP10453"

# --- Files classified as personal/sensitive vs. general -------------------
# Used by ingest.py to tag each chunk's metadata. In a real pipeline this
# classification might come from a document management system instead of
# being hardcoded, but for 8 known sample files this is simplest.
PERSONAL_DOCS = {
    "03_offer_letter_sample.md": "EMP10453",
    "04_appraisal_letter_sample.md": "EMP10453",
}
