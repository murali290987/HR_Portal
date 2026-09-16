"""
query.py — Step 4: retrieval + generation.

Takes a question from the command line, embeds it with the same model
used during ingestion, retrieves the top-k most similar chunks from the
FAISS index built by ingest.py, and asks an LLM to answer using ONLY
those chunks as context (this is the "Retrieval-Augmented" part of RAG:
we augment the LLM's own knowledge with facts pulled from your specific
documents, instead of trusting whatever it remembers from training).

Usage:
    python query.py "How many casual leaves do I get per year?"
"""

import json
import re
import sys

import faiss
from sentence_transformers import SentenceTransformer

import config


def load_index_and_metadata(index_path=None, metadata_path=None):
    """
    Load the FAISS index and its paired metadata list that ingest.py
    built. These two files only make sense together: FAISS only knows
    "vector at position i is the closest match" — the metadata list is
    what turns position i back into actual chunk text, source_file,
    doc_type, and employee_id.

    index_path/metadata_path default to config.py's paths -- pass them
    explicitly (e.g. from evals/sweep.py) to load a throwaway index
    built from a temp directory instead of the real one.
    """
    index_path = index_path if index_path is not None else config.FAISS_INDEX_PATH
    metadata_path = metadata_path if metadata_path is not None else config.METADATA_PATH

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def _query_names_employee(query_text: str, employee_id: str) -> bool:
    """
    Does the query explicitly name this specific employee, by id or by
    name? Used by retrieve()'s hr-role subject check below -- the hr
    role's ownership bypass exists so HR can look up a NAMED employee's
    records, not so any vague personal question ("what is my name?")
    resolves to whichever employee's document happens to be closest.
    """
    mentioned_ids = {m.upper() for m in _EMPLOYEE_ID_RE.findall(query_text)}
    if employee_id in mentioned_ids:
        return True
    name = config.EMPLOYEE_NAMES.get(employee_id)
    return name is not None and name.lower() in query_text.lower()


def retrieve(
    query: str,
    model: SentenceTransformer,
    index,
    metadata: list,
    top_k: int,
    current_user_id: str,
    current_role: str,
) -> list:
    """
    Embed the query with the SAME model used to embed the chunks during
    ingestion — this matters, because a query embedded with a different
    model would land in a differently-shaped vector space, making
    distance comparisons meaningless.

    Step 5 — naive access control (now RBAC-extended):
    In a real system, `current_user_id` and `current_role` would come
    from a verified, authenticated session (e.g. a checked JWT's
    `employee_id` and `role` claims — recall the JWT demo payload had a
    "role" field that went unused until now), and access rules would be
    enforced server-side per request. Here we just take them as plain
    arguments so we can see the *filtering logic* work — no real AuthN
    (proving who you are) is happening; that's a later phase.

    The filter: a chunk tagged "personal" belonging to someone other
    than current_user_id is excluded UNLESS current_role is "hr" --
    matching what a real HR admin would need. But the hr bypass only
    applies when the query actually names whose record it wants (by id
    or by name, via _query_names_employee): an hr request for a vague
    "my ..." question names no one, so it should not silently resolve
    to whichever employee's chunk happens to be closest. This was a
    real bug -- HR001 asking "what is my name" got back "Priya
    Ramanathan" -- fixed here rather than in apply_relevance_guardrail()
    because it's about which chunks are ELIGIBLE at all, not about
    re-filtering chunks that already passed access control. General docs
    are never filtered by any of this — they're not employee-specific.

    Implementation note: we ask FAISS to rank ALL chunks (not just
    top_k), apply the filter to that full ranked list, and only THEN
    take the top_k survivors. If we filtered *after* asking FAISS for
    just top_k, a blocked personal chunk could occupy one of the top_k
    slots and silently crowd out a general chunk that should have been
    returned instead — ranking everything first and filtering before
    truncating avoids that.
    """
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    total_chunks = index.ntotal
    distances, positions = index.search(query_vector, total_chunks)

    results = []
    for distance, position in zip(distances[0], positions[0]):
        chunk = metadata[position]
        belongs_to_someone_else = (
            chunk["doc_type"] == "personal" and chunk["employee_id"] != current_user_id
        )
        if belongs_to_someone_else:
            if current_role != "hr":
                continue  # no bypass available -- blocked by ownership
            if not _query_names_employee(query, chunk["employee_id"]):
                continue  # hr bypass requires naming whose record this is
        results.append({**chunk, "distance": float(distance)})
        if len(results) == top_k:
            break
    return results


NO_RELEVANT_CONTEXT_MESSAGE = (
    "I don't have relevant information in the available documents to answer that question."
)

_EMPLOYEE_ID_RE = re.compile(config.EMPLOYEE_ID_PATTERN, re.IGNORECASE)


def apply_relevance_guardrail(query_text: str, retrieved_chunks: list) -> list:
    """
    A relevance guardrail, distinct from access control: access control
    (Step 5) only checks WHO is allowed to see a chunk; this checks
    whether a chunk the requester IS allowed to see is actually about
    who/what the question is asking about. A chunk can pass access
    control and still be the wrong answer — e.g. you (EMP10453) asking
    about a DIFFERENT employee's CTC ("EMP99999"). Your own offer letter
    is yours to see, but it doesn't answer a question about someone else
    — handing it to the LLM anyway is what caused the misattribution bug
    (the model just grabbed the CTC number in front of it and pinned it
    on whichever ID the question mentioned).

    The check: if the question explicitly names an employee ID, any
    personal chunk belonging to a DIFFERENT employee_id gets dropped,
    regardless of how "close" its embedding distance is. Questions that
    don't name a specific ID (e.g. "What is my CTC?") are left untouched
    — this check only fires on the exact pattern that caused the bug.

    We deliberately do NOT use a blanket embedding-distance cutoff for
    the general "nothing here is relevant" case — see config.py's
    comment on why that was tried and abandoned. That broader case is
    instead left to the prompt's existing instruction to say "I don't
    know" when the context doesn't answer the question, which already
    works correctly for clearly off-topic questions (e.g. asking about
    stock options when nothing in the docs mentions them).
    """
    mentioned_ids = {m.upper() for m in _EMPLOYEE_ID_RE.findall(query_text)}
    if not mentioned_ids:
        return retrieved_chunks

    return [
        chunk
        for chunk in retrieved_chunks
        if not (chunk["doc_type"] == "personal" and chunk["employee_id"] not in mentioned_ids)
    ]


def build_prompt(query: str, retrieved_chunks: list) -> str:
    """
    Construct the prompt sent to the LLM. This is the crux of RAG:
    instead of asking the LLM to answer purely from what it memorized
    during training (which knows nothing about YOUR company's specific
    leave policy), we hand it the retrieved chunks as context and
    instruct it to answer ONLY from that context. That's what keeps the
    answer grounded in your actual documents instead of the model
    guessing — and what lets it correctly say "I don't know" when the
    retrieved chunks don't actually answer the question.
    """
    context_blocks = [
        f"[Source: {chunk['source_file']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ]
    context = "\n\n".join(context_blocks)

    return f"""You are an HR assistant answering employee questions using ONLY the context provided below.

Context:
{context}

Question: {query}

Instructions:
- Answer using only the information in the context above.
- If the context does not contain enough information to answer the question, say so clearly instead of guessing. Do not make anything up.
- Keep the answer concise and direct.

Answer:"""


def call_anthropic(prompt: str) -> str:
    """Send the prompt to Claude via the Anthropic API. Needs ANTHROPIC_API_KEY in .env."""
    import os

    import anthropic
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key, "
            "or switch config.LLM_PROVIDER to 'ollama'."
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_ollama(prompt: str) -> str:
    """Send the prompt to a locally running Ollama model. No API key needed."""
    import ollama

    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def generate_answer(prompt: str, provider: str = None) -> str:
    """
    Dispatch to whichever backend is selected. Same prompt either way.

    `provider` defaults to config.LLM_PROVIDER (what the CLI uses), but
    callers like server.py can pass a per-request override instead —
    e.g. a dropdown in a UI letting someone pick Anthropic vs Ollama
    without touching config.py.
    """
    provider = provider or config.LLM_PROVIDER
    if provider == "anthropic":
        return call_anthropic(prompt)
    elif provider == "ollama":
        return call_ollama(prompt)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}")
    # Step 5 (naive access control, RBAC-extended): stands in for
    # "whoever is logged in right now." Change CURRENT_USER_ID in
    # config.py (or add a new entry to USERS) and re-run to see personal
    # docs get included/excluded from retrieval accordingly.
    current_role = config.USERS[config.CURRENT_USER_ID]["role"]
    print(f"Current user: {config.CURRENT_USER_ID} (role: {current_role})")

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    index, metadata = load_index_and_metadata()

    retrieved = retrieve(
        query, model, index, metadata, config.TOP_K, config.CURRENT_USER_ID, current_role
    )

    print(f"\n--- Retrieved top {config.TOP_K} chunks ---")
    for i, chunk in enumerate(retrieved):
        print(
            f"{i + 1}. [{chunk['source_file']}] distance={chunk['distance']:.3f} "
            f"doc_type={chunk['doc_type']} employee_id={chunk['employee_id']}"
        )
        print(f"   {chunk['text'][:150]!r}")

    relevant = apply_relevance_guardrail(query, retrieved)
    if len(relevant) != len(retrieved):
        print(f"\nRelevance guardrail dropped {len(retrieved) - len(relevant)} chunk(s)")

    if not relevant:
        # Guardrail: don't even call the LLM if nothing retrieved is
        # actually relevant -- makes "I don't know" deterministic rather
        # than depending on the LLM noticing the context doesn't apply.
        answer = NO_RELEVANT_CONTEXT_MESSAGE
    else:
        prompt = build_prompt(query, relevant)
        print(f"\n--- Generating answer via {config.LLM_PROVIDER} ({config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else config.ANTHROPIC_MODEL}) ---")
        answer = generate_answer(prompt)

    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
