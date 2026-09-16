"""
evals/run_retrieval_eval.py — Stage 2: retrieval-only eval, NO LLM calls.

Runs every case in dataset.py through the real, unmodified retrieval
path -- query.retrieve() then query.apply_relevance_guardrail(), the
exact same two functions query.py's CLI and server.py's API call. This
file does not reimplement either of them; it only measures what they
produce.

Deliberately fast: retrieval quality doesn't depend on which LLM writes
the final sentence, so this should be cheap enough to run after every
config.py tweak. The slow, LLM-calling version is run_answer_eval.py
(Stage 4), run on demand instead.

Run:
    ./.venv/bin/python3 evals/run_retrieval_eval.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so `import config` / `import query` resolve from hr_rag/

import config
import query
from sentence_transformers import SentenceTransformer

from dataset import CASES

RESULTS_DIR = Path(__file__).parent / "results"


# --- Metric definitions ---------------------------------------------------
#
# All four metrics below are computed on `relevant` -- the chunk list
# AFTER apply_relevance_guardrail() has run, i.e. exactly what would
# actually be pasted into the LLM prompt in production (server.py builds
# the prompt from this same post-guardrail list, never from the raw
# retrieve() output). Measuring the raw `retrieved` list instead would
# make guardrail-blocked cases look like retrieval failures when they're
# actually guardrail successes -- e.g. other_id_named_as_owner SHOULD
# show zero of everything here, because the guardrail correctly emptied
# the list before anything reached the LLM.
#
# hit@k: did at least one of expected_sources actually appear? This is
# the coarsest, most forgiving metric -- it's satisfied even if the
# right document is buried behind two irrelevant ones. Cases with an
# empty expected_sources (no_match cases) have no "hit" to check, so
# they're marked "n/a" and excluded from the category average rather
# than silently counted as a free pass.
#
# precision@k: (chunks whose source_file is in expected_sources) /
# (chunks actually returned). This is the metric most likely to look
# "bad" even when the answer would come out fine -- e.g. if expected
# top-3 needs 2 chunks from the right doc and top_k=3 fills the 3rd slot
# with something unrelated-but-closer-by-distance, precision@k = 2/3
# even though the LLM still had everything it needed. Low precision@k
# is a sign of noisy retrieval, not necessarily a wrong final answer --
# that's exactly why Stage 4 checks answer content separately rather
# than inferring quality from this number alone.
#
# recall (on expected_sources): (expected_sources files actually
# represented) / (total expected_sources files). Unlike precision, this
# one really does matter on its own -- if a required source never shows
# up at all, the LLM structurally cannot have used it, whatever its
# distance-based excuse.
#
# leak count: chunks whose source_file is in forbidden_sources. This is
# the one hard rule -- ANY leak, even one chunk, is a correctness bug,
# not a tuning tradeoff, and is reported separately from the quality
# metrics above rather than averaged into them.
def compute_case_metrics(case, relevant_chunks):
    sources = [c["source_file"] for c in relevant_chunks]
    source_set = set(sources)

    expected = set(case["expected_sources"])
    forbidden = set(case["forbidden_sources"])

    hit_at_k = (len(expected & source_set) > 0) if expected else None
    precision_at_k = (
        sum(1 for s in sources if s in expected) / len(sources) if sources and expected else None
    )
    recall = (len(expected & source_set) / len(expected)) if expected else None
    leaked_sources = sorted(source_set & forbidden)

    return {
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "recall": recall,
        "leak_count": len(leaked_sources),
        "leaked_sources": leaked_sources,
        "retrieved_sources": sources,
    }


def run_case(case, model, index, metadata, top_k=None):
    """
    top_k defaults to config.TOP_K -- sweep.py passes it explicitly so
    the exact same per-case logic can be reused across the TOP_K grid
    without duplicating it.
    """
    top_k = top_k if top_k is not None else config.TOP_K
    role = config.USERS[case["user_id"]]["role"]

    retrieved = query.retrieve(
        case["question"], model, index, metadata, top_k, case["user_id"], role
    )
    relevant = query.apply_relevance_guardrail(case["question"], retrieved)
    guardrail_triggered = len(relevant) == 0

    metrics = compute_case_metrics(case, relevant)
    metrics["guardrail_triggered"] = guardrail_triggered
    metrics["guardrail_match"] = guardrail_triggered == case["expect_guardrail"]
    return metrics


def fmt_pct(value):
    return "n/a" if value is None else f"{value * 100:.0f}%"


def print_report(all_results):
    regular = [r for r in all_results if not r["case"]["known_limitation"]]
    known_limitation = [r for r in all_results if r["case"]["known_limitation"]]

    print("=" * 78)
    print("RETRIEVAL EVAL -- regular cases")
    print("=" * 78)
    _print_table(regular)

    total_leaks = sum(r["metrics"]["leak_count"] for r in regular)
    if total_leaks:
        # Split by leak_type rather than reporting one flat count: an
        # access_violation (requester has no legitimate claim to this
        # data at all) is a different, more severe bug than a
        # wrong_subject leak (permissions were correct, the question
        # was just about the wrong person) -- see dataset.py's field
        # docs. Different layers, different fixes.
        access_violation_leaks = sum(
            r["metrics"]["leak_count"] for r in regular if r["case"]["leak_type"] == "access_violation"
        )
        wrong_subject_leaks = sum(
            r["metrics"]["leak_count"] for r in regular if r["case"]["leak_type"] == "wrong_subject"
        )
        print(f"\n*** {total_leaks} LEAK(S) DETECTED in non-known-limitation cases. ***")
        print(f"    access_violation: {access_violation_leaks}  |  wrong_subject: {wrong_subject_leaks}")
        for r in regular:
            if r["metrics"]["leak_count"]:
                print(
                    f"    [{r['case']['id']}] ({r['case']['leak_type']}) "
                    f"leaked sources: {r['metrics']['leaked_sources']}"
                )
    else:
        print("\nNo leaks in regular cases.")

    guardrail_mismatches = [r for r in regular if not r["metrics"]["guardrail_match"]]
    if guardrail_mismatches:
        print(f"\n*** {len(guardrail_mismatches)} GUARDRAIL MISMATCH(ES): ***")
        for r in guardrail_mismatches:
            print(
                f"    [{r['case']['id']}] expected guardrail={r['case']['expect_guardrail']}, "
                f"got {r['metrics']['guardrail_triggered']}"
            )

    if known_limitation:
        print("\n" + "=" * 78)
        print("KNOWN-LIMITATION cases (expected to fail today -- not counted above)")
        print("=" * 78)
        _print_table(known_limitation)
        for r in known_limitation:
            status = "still failing as expected" if r["metrics"]["leak_count"] else "NOW PASSING -- update dataset.py"
            print(f"    [{r['case']['id']}] ({r['case']['leak_type']}) {status}")


def _print_table(results):
    if not results:
        print("(none)")
        return

    by_category = defaultdict(list)
    for r in results:
        by_category[r["case"]["category"]].append(r)

    header = f"{'category':<20}{'n':<4}{'hit@k':<8}{'precision@k':<13}{'recall':<8}{'leaks':<7}{'guardrail_ok':<13}"
    print(header)
    print("-" * len(header))

    for category, rows in sorted(by_category.items()):
        hits = [r["metrics"]["hit_at_k"] for r in rows if r["metrics"]["hit_at_k"] is not None]
        precisions = [r["metrics"]["precision_at_k"] for r in rows if r["metrics"]["precision_at_k"] is not None]
        recalls = [r["metrics"]["recall"] for r in rows if r["metrics"]["recall"] is not None]
        leaks = sum(r["metrics"]["leak_count"] for r in rows)
        guardrail_ok = sum(1 for r in rows if r["metrics"]["guardrail_match"])

        hit_avg = sum(hits) / len(hits) if hits else None
        prec_avg = sum(precisions) / len(precisions) if precisions else None
        recall_avg = sum(recalls) / len(recalls) if recalls else None

        print(
            f"{category:<20}{len(rows):<4}{fmt_pct(hit_avg):<8}{fmt_pct(prec_avg):<13}"
            f"{fmt_pct(recall_avg):<8}{leaks:<7}{guardrail_ok}/{len(rows):<11}"
        )

    all_hits = [r["metrics"]["hit_at_k"] for r in results if r["metrics"]["hit_at_k"] is not None]
    all_guardrail_ok = sum(1 for r in results if r["metrics"]["guardrail_match"])
    overall_hit = sum(all_hits) / len(all_hits) if all_hits else None
    print("-" * len(header))
    print(
        f"{'OVERALL':<20}{len(results):<4}{fmt_pct(overall_hit):<8}"
        f"{'':<13}{'':<8}{sum(r['metrics']['leak_count'] for r in results):<7}"
        f"{all_guardrail_ok}/{len(results)}"
    )


def main():
    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    index, metadata = query.load_index_and_metadata()

    all_results = []
    for case in CASES:
        metrics = run_case(case, model, index, metadata)
        all_results.append({"case": case, "metrics": metrics})

    print_report(all_results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"retrieval_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"case": r["case"], "metrics": r["metrics"]} for r in all_results],
            f,
            indent=2,
        )
    print(f"\nFull per-case results written to {out_path}")


if __name__ == "__main__":
    main()
