"""
evals/sweep.py — Stage 3: sweep chunking/retrieval config and compare.

Runs Stage 2's exact per-case eval logic (run_retrieval_eval.run_case,
imported and reused, not reimplemented) across a grid of:

    CHUNK_SIZE    in [300, 500, 800, 1200]
    CHUNK_OVERLAP in [0, 50, 150]
    TOP_K         in [1, 3, 5]

and prints a comparison table sorted by overall hit@k, flagging any
configuration that produces a leak.

Why the grid is structured as (chunk_size, overlap) outer loop, top_k
inner loop: changing chunk_size/overlap changes what a "chunk" even IS,
so it requires re-running ingestion from scratch -- new chunks, new
embeddings, a new FAISS index. That's the expensive step. Changing
top_k does NOT require re-ingestion -- it only changes how many results
retrieve() pulls back from an ALREADY-BUILT index. So this sweep
re-ingests once per (chunk_size, overlap) pair (12 times for the
default grid) and, for each, re-runs the 20-case eval once per top_k
value (3x) against that same index -- 36 total eval passes, but only 12
embedding passes.

Every ingestion here writes to a fresh temp directory (via
tempfile.TemporaryDirectory), never to config.FAISS_INDEX_PATH /
config.METADATA_PATH -- your real index, and query.py/server.py's
runtime behavior, are completely untouched by running this script. The
temp directory is deleted automatically when each iteration's `with`
block exits.

Run:
    ./.venv/bin/python3 evals/sweep.py
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import ingest
import query
from sentence_transformers import SentenceTransformer

from dataset import CASES
from run_retrieval_eval import run_case

CHUNK_SIZE_GRID = [300, 500, 800, 1200]
CHUNK_OVERLAP_GRID = [0, 50, 150]
TOP_K_GRID = [1, 3, 5]


def build_temp_index(chunk_size, overlap, model, tmp_dir):
    """Re-ingest with the given chunk_size/overlap into a throwaway tmp_dir."""
    chunks = ingest.build_chunks(config.HR_DOCS_DIR, chunk_size=chunk_size, overlap=overlap)
    embeddings = ingest.embed_chunks(chunks, model)
    index = ingest.build_faiss_index(embeddings)

    index_path = tmp_dir / "faiss_index.bin"
    metadata_path = tmp_dir / "chunks_metadata.json"
    ingest.save_index_and_metadata(index, chunks, index_path=index_path, metadata_path=metadata_path)

    return query.load_index_and_metadata(index_path=index_path, metadata_path=metadata_path)


def summarize(results):
    hits = [r["metrics"]["hit_at_k"] for r in results if r["metrics"]["hit_at_k"] is not None]
    leaks_regular = sum(r["metrics"]["leak_count"] for r in results if not r["case"]["known_limitation"])
    leaks_known = sum(r["metrics"]["leak_count"] for r in results if r["case"]["known_limitation"])
    guardrail_ok = sum(1 for r in results if r["metrics"]["guardrail_match"])
    return {
        "hit_rate": (sum(hits) / len(hits)) if hits else 0.0,
        "leaks_regular": leaks_regular,
        "leaks_known": leaks_known,
        "guardrail_ok": guardrail_ok,
        "n": len(results),
    }


def print_comparison_table(grid_results):
    grid_results = sorted(grid_results, key=lambda r: r["hit_rate"], reverse=True)

    header = (
        f"{'chunk_size':<12}{'overlap':<9}{'top_k':<7}{'hit@k':<8}"
        f"{'leaks':<7}{'known_leaks':<13}{'guardrail_ok':<13}"
    )
    print("\n" + "=" * len(header))
    print("SWEEP RESULTS (sorted by overall hit@k, best first)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in grid_results:
        marker = "  <-- LEAK" if r["leaks_regular"] > 0 else ""
        print(
            f"{r['chunk_size']:<12}{r['overlap']:<9}{r['top_k']:<7}"
            f"{r['hit_rate'] * 100:>4.0f}%   {r['leaks_regular']:<7}{r['leaks_known']:<13}"
            f"{r['guardrail_ok']}/{r['n']:<11}{marker}"
        )

    leaking = [r for r in grid_results if r["leaks_regular"] > 0]
    if leaking:
        print(f"\n*** {len(leaking)} configuration(s) produced a LEAK in non-known-limitation cases. ***")
        print("A leak caused purely by a chunking/top_k change is a serious finding -- investigate before tuning further.")
        for r in leaking:
            print(f"    chunk_size={r['chunk_size']} overlap={r['overlap']} top_k={r['top_k']}: {r['leaks_regular']} leak(s)")
    else:
        print("\nNo configuration produced a leak in non-known-limitation cases.")

    known_leak_values = {r["leaks_known"] for r in grid_results}
    if len(known_leak_values) == 1:
        print(
            f"\nKnown-limitation case leak count is constant at "
            f"{known_leak_values.pop()} across every configuration tested."
        )
    else:
        print(
            f"\nKnown-limitation case leak count VARIES by configuration "
            f"(values seen: {sorted(known_leak_values)}) -- chunking does "
            f"affect how much of the personal doc surfaces for that case; "
            f"see the per-row breakdown above."
        )


def main():
    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    valid_pairs = [(cs, ov) for cs in CHUNK_SIZE_GRID for ov in CHUNK_OVERLAP_GRID if ov < cs]
    skipped = [(cs, ov) for cs in CHUNK_SIZE_GRID for ov in CHUNK_OVERLAP_GRID if ov >= cs]
    if skipped:
        print(f"Skipping invalid (chunk_size, overlap) pairs where overlap >= chunk_size: {skipped}")

    grid_results = []
    for chunk_size, overlap in valid_pairs:
        print(f"Re-ingesting: chunk_size={chunk_size}, overlap={overlap} ...")
        with tempfile.TemporaryDirectory(prefix="hr_rag_sweep_") as tmp:
            tmp_dir = Path(tmp)
            index, metadata = build_temp_index(chunk_size, overlap, model, tmp_dir)

            for top_k in TOP_K_GRID:
                results = [
                    {"case": case, "metrics": run_case(case, model, index, metadata, top_k=top_k)}
                    for case in CASES
                ]
                grid_results.append(
                    {"chunk_size": chunk_size, "overlap": overlap, "top_k": top_k, **summarize(results)}
                )

    print_comparison_table(grid_results)


if __name__ == "__main__":
    main()
