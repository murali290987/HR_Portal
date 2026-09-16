"""
evals/test_regressions.py — Stage 5: pytest wrapper around Stage 2.

Runs the same retrieval-only eval as run_retrieval_eval.py (reusing its
run_case() directly, not reimplementing it) -- no LLM calls, so this is
fast enough to run on every commit. Fails the build on:

  - ANY leak in a non-known-limitation case (a forbidden_sources file
    actually appearing in the retrieved set). Zero tolerance -- this is
    a correctness bug, not a tuning tradeoff, exactly as Stage 2 treats it.
  - Overall hit@k across non-known-limitation cases dropping below
    MIN_HIT_RATE, set once below.

known_limitation cases are excluded entirely from both checks -- not
"allowed to fail," but not evaluated as pass/fail at all. They're an
accepted, tracked, pre-existing gap (see dataset.py), not a regression
this test suite should ever fail the build over.

Run:
    ./.venv/bin/python3 -m pytest evals/test_regressions.py -v -s
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import query
from sentence_transformers import SentenceTransformer

from dataset import CASES
from run_retrieval_eval import run_case

# Ratcheted to 1.0 after fixing the HR first-person bypass and the
# chunker tail-merge bug (see evals/FINDINGS.md) -- re-measured hit@k
# across all applicable regular cases is genuinely 100% now, so a floor
# below that would tolerate a regression rather than catch one. Lower
# it only for a real, understood reason -- never just to silence a
# failure.
#
# IMPORTANT LIMITATION: this floor does NOT cover the still-open
# appraisal-letter chunking gap (the revised-CTC chunk misses top-3).
# hit@k is file-level -- "did 04_appraisal_letter_sample.md show up
# ANYWHERE" -- and it does (via the header chunk), so hit@k reads True
# for hr_full_appraisal_lookup even though the specific chunk with the
# actual number is missing. Only Stage 4's expected_answer_contains
# check catches that, and Stage 4 (LLM calls) isn't part of this pytest
# gate. Don't read "1.0 hit@k, gate passes" as "no known gaps remain."
MIN_HIT_RATE = 1.0


@pytest.fixture(scope="session")
def pipeline():
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    index, metadata = query.load_index_and_metadata()
    return model, index, metadata


@pytest.fixture(scope="session")
def all_results(pipeline):
    model, index, metadata = pipeline
    return [{"case": case, "metrics": run_case(case, model, index, metadata)} for case in CASES]


@pytest.fixture(scope="session")
def regular_results(all_results):
    return [r for r in all_results if not r["case"]["known_limitation"]]


def test_no_leaks_in_regular_cases(regular_results):
    """
    Zero tolerance regardless of leak_type -- access_violation and
    wrong_subject are different severities/layers (see dataset.py), but
    both are correctness bugs, so both fail the build the same way here.
    The type is included in the assertion message purely so a failure is
    faster to triage, not because one type is allowed to pass.
    """
    leaking = [
        (r["case"]["id"], r["case"]["leak_type"], r["metrics"]["leaked_sources"])
        for r in regular_results
        if r["metrics"]["leak_count"] > 0
    ]
    assert not leaking, f"Leak(s) detected in non-known-limitation cases (id, leak_type, sources): {leaking}"


def test_hit_rate_meets_floor(regular_results):
    hits = [r["metrics"]["hit_at_k"] for r in regular_results if r["metrics"]["hit_at_k"] is not None]
    hit_rate = (sum(hits) / len(hits)) if hits else 1.0
    assert hit_rate >= MIN_HIT_RATE, (
        f"Overall hit@k {hit_rate:.0%} is below the floor of {MIN_HIT_RATE:.0%} "
        f"({sum(hits)}/{len(hits)} applicable cases hit)"
    )


def test_known_limitation_cases_are_tracked(all_results):
    """
    Not a quality gate -- just a guard against dataset.py silently
    losing its known_limitation case (e.g. someone deletes it thinking
    it's noise). If this fails, the tracked gap stopped being tracked.
    """
    known = [r for r in all_results if r["case"]["known_limitation"]]
    assert len(known) >= 1, "Expected at least one known_limitation case, found none."
