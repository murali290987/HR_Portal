"""
evals/run_answer_eval.py — Stage 4: answer-quality eval, WITH LLM calls.

Unlike run_retrieval_eval.py (fast, no LLM calls, safe to run after
every tweak), this runs the FULL pipeline -- including a real call to
generate_answer() -- for every case except those with
expect_guardrail=True. Those are skipped entirely: in production, the
guardrail returns a fixed message with no LLM call at all, so there's
nothing new to check that Stage 2 doesn't already verify. This makes
Stage 4 slow and dependent on whichever provider you pick -- run it on
demand, not routinely.

For each case that runs:
  1. retrieve() -> apply_relevance_guardrail() -> build_prompt() ->
     generate_answer(), the exact same functions query.py's CLI and
     server.py use. Nothing reimplemented.
  2. Deterministic fact check: does the answer contain every string in
     expected_answer_contains (case-insensitive substring match)? Empty
     for no_match cases -- see looks_like_no_answer() instead.
  3. no_match cases only: does the answer look like a refusal rather
     than a fabricated fact? See looks_like_no_answer()'s docstring for
     exactly how, and where it can be wrong.
  4. Content-level leak check: does the answer contain a fingerprint
     string unique to a forbidden_sources document? This complements
     Stage 2's retrieval-level leak check (which looks at what got
     RETRIEVED) by looking at what actually made it into the final
     ANSWER TEXT -- the thing an end user actually sees.

Usage:
    ./.venv/bin/python3 evals/run_answer_eval.py --provider ollama
    ./.venv/bin/python3 evals/run_answer_eval.py --provider anthropic
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import query
from sentence_transformers import SentenceTransformer

from dataset import CASES

RESULTS_DIR = Path(__file__).parent / "results"

# Distinctive strings that should only plausibly appear in an answer if
# content from that specific document leaked into it. Deliberately
# specific numbers/phrases pulled from the actual doc text, not generic
# words, to avoid false positives against legitimate general-policy
# content that happens to share a word with a personal document.
DOC_FINGERPRINTS = {
    "03_offer_letter_sample.md": ["14,50,000", "5,80,000", "05-feb-2024", "priya ramanathan"],
    "04_appraisal_letter_sample.md": ["16,90,000", "16.5%", "senior software engineer", "priya ramanathan"],
}


# --- "Does this look like a refusal?" detector ----------------------------
#
# Purely lexical: lowercases the answer and checks for common refusal
# phrasings. Known weaknesses (worth knowing before trusting this):
#   - False negative risk: a model can refuse in a phrasing this list
#     doesn't cover (e.g. "I'm not able to help with that") and get
#     wrongly marked as NOT a refusal.
#   - False positive risk: an answer that legitimately contains one of
#     these phrases as part of an otherwise-real answer (e.g. "There is
#     no cap mentioned beyond X") could be wrongly marked as a refusal.
#   - It is not a factuality check. This only detects refusal-shaped
#     TEXT -- it says nothing about whether a non-refusal answer is
#     actually correct. That's what expected_answer_contains is for.
#   - No semantic understanding at all -- a refusal phrased in
#     completely different words, or in a language other than English,
#     would be missed entirely.
NO_ANSWER_PHRASES = [
    "don't have", "do not have", "not mentioned", "no information",
    "not have enough information", "insufficient information",
    "cannot find", "can't find", "not provided", "not contain",
    "doesn't contain", "does not contain", "isn't mentioned",
    "not specified", "unable to find", "not aware of",
]


def looks_like_no_answer(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in NO_ANSWER_PHRASES)


def check_content_leak(answer_text: str, forbidden_sources: list) -> list:
    """Returns [(source_file, fingerprint), ...] for every forbidden fingerprint found in the answer."""
    lowered = answer_text.lower()
    leaked = []
    for source in forbidden_sources:
        for fingerprint in DOC_FINGERPRINTS.get(source, []):
            if fingerprint.lower() in lowered:
                leaked.append((source, fingerprint))
    return leaked


def run_case(case, model, index, metadata, provider):
    role = config.USERS[case["user_id"]]["role"]
    retrieved = query.retrieve(
        case["question"], model, index, metadata, config.TOP_K, case["user_id"], role
    )
    relevant = query.apply_relevance_guardrail(case["question"], retrieved)
    guardrail_triggered = len(relevant) == 0

    if guardrail_triggered:
        # Shouldn't happen for a non-expect_guardrail=True case at
        # default config, but if it does, mirror production exactly:
        # fixed message, no LLM call.
        answer = query.NO_RELEVANT_CONTEXT_MESSAGE
    else:
        prompt = query.build_prompt(case["question"], relevant)
        answer = query.generate_answer(prompt, provider=provider)

    lowered_answer = answer.lower()
    missing_facts = [
        fact for fact in case["expected_answer_contains"] if fact.lower() not in lowered_answer
    ]
    facts_ok = (len(missing_facts) == 0) if case["expected_answer_contains"] else None

    no_answer_check = looks_like_no_answer(answer) if case["category"] == "no_match" else None

    content_leaks = check_content_leak(answer, case["forbidden_sources"])

    return {
        "answer": answer,
        "guardrail_triggered": guardrail_triggered,
        "missing_facts": missing_facts,
        "facts_ok": facts_ok,
        "no_answer_check": no_answer_check,
        "content_leaks": content_leaks,
    }


def print_report(all_results):
    regular = [r for r in all_results if not r["case"]["known_limitation"]]
    known_limitation = [r for r in all_results if r["case"]["known_limitation"]]

    print("=" * 90)
    print("ANSWER EVAL -- regular cases")
    print("=" * 90)
    for r in regular:
        _print_case(r)

    fact_checked = [r for r in regular if r["result"]["facts_ok"] is not None]
    facts_passed = sum(1 for r in fact_checked if r["result"]["facts_ok"])
    no_match_checked = [r for r in regular if r["result"]["no_answer_check"] is not None]
    no_match_passed = sum(1 for r in no_match_checked if r["result"]["no_answer_check"])
    total_leaks = sum(len(r["result"]["content_leaks"]) for r in regular)

    print("-" * 90)
    print(f"Fact checks:     {facts_passed}/{len(fact_checked)} passed")
    print(f"No-match checks: {no_match_passed}/{len(no_match_checked)} looked like a refusal")
    if total_leaks:
        print(f"\n*** {total_leaks} CONTENT-LEVEL LEAK(S) in regular-case answers: ***")
        for r in regular:
            if r["result"]["content_leaks"]:
                print(f"    [{r['case']['id']}] leaked: {r['result']['content_leaks']}")
    else:
        print("Content-level leaks: 0")

    if known_limitation:
        print("\n" + "=" * 90)
        print("KNOWN-LIMITATION cases (expected to fail today -- not counted above)")
        print("=" * 90)
        for r in known_limitation:
            _print_case(r)
            leaks = r["result"]["content_leaks"]
            status = f"still leaking as expected: {leaks}" if leaks else "NOW CLEAN -- update dataset.py"
            print(f"    -> {status}")


def _print_case(r):
    case, result = r["case"], r["result"]
    print(f"\n[{case['id']}] ({case['category']}, user={case['user_id']})")
    print(f"  Q: {case['question']}")
    print(f"  A: {result['answer'][:200]!r}{'...' if len(result['answer']) > 200 else ''}")
    if result["facts_ok"] is not None:
        status = "OK" if result["facts_ok"] else f"MISSING: {result['missing_facts']}"
        print(f"  facts_ok: {status}")
    if result["no_answer_check"] is not None:
        print(f"  looks_like_no_answer: {result['no_answer_check']}")
    if result["content_leaks"]:
        print(f"  *** CONTENT LEAK: {result['content_leaks']} ***")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["ollama", "anthropic"], default=config.LLM_PROVIDER)
    args = parser.parse_args()

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    index, metadata = query.load_index_and_metadata()

    runnable_cases = [c for c in CASES if not c["expect_guardrail"]]
    skipped = [c["id"] for c in CASES if c["expect_guardrail"]]
    if skipped:
        print(f"Skipping expect_guardrail=True cases (no LLM call happens for these in production): {skipped}")

    print(f"Running {len(runnable_cases)} cases via provider={args.provider} ...")

    all_results = []
    for case in runnable_cases:
        result = run_case(case, model, index, metadata, args.provider)
        all_results.append({"case": case, "result": result})

    print_report(all_results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"answer_{args.provider}_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"case": r["case"], "result": r["result"]} for r in all_results],
            f,
            indent=2,
        )
    print(f"\nFull per-case results written to {out_path}")


if __name__ == "__main__":
    main()
