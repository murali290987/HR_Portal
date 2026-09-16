# Eval harness

A golden-dataset eval harness for measuring whether a change (chunk
size, `TOP_K`, prompt wording) actually helped retrieval/answer quality,
instead of eyeballing a handful of queries.

## Files, and when to run each

| File | What it does | LLM calls? | When to run |
|---|---|---|---|
| `dataset.py` | 20 hand-built test cases across 7 categories, grounded in `hr_docs/`'s actual content. Not a script you run — imported by everything below. | No | N/A |
| `run_retrieval_eval.py` | Runs every case through the real `query.retrieve()` + `query.apply_relevance_guardrail()`. Reports hit@k, precision@k, recall, leak count, guardrail accuracy. | No | After any change to `config.py`, `ingest.py`, or `query.py`'s retrieval logic — fast enough for every run. |
| `sweep.py` | Runs `run_retrieval_eval`'s per-case logic across a grid of `CHUNK_SIZE` × `CHUNK_OVERLAP` × `TOP_K` (36 combinations by default), each ingested into a throwaway temp directory — your real `faiss_index.bin` is never touched. | No | When actually deciding what to set `CHUNK_SIZE`/`CHUNK_OVERLAP`/`TOP_K` to. Slower than `run_retrieval_eval.py` (re-ingests up to 12 times) but still no LLM calls. |
| `run_answer_eval.py` | Runs the full pipeline (retrieval + generation) for every case except `expect_guardrail: True` ones. Checks `expected_answer_contains` substrings, a lexical "does this look like a refusal" heuristic for `no_match` cases, and a content-level leak check on the final answer text. | **Yes** | On demand — to check whether the LLM itself (not just retrieval) is getting answers right, or to compare `--provider ollama` vs `--provider anthropic`. Slow, costs real inference time. |
| `test_regressions.py` | pytest wrapper around `run_retrieval_eval`'s logic. Fails the build on any leak, or on overall hit@k dropping below `MIN_HIT_RATE` (one constant, top of the file). | No | In CI / before committing a retrieval-affecting change. |

## Running things

```bash
# Fast retrieval-only eval — run this often
./.venv/bin/python3 evals/run_retrieval_eval.py

# Config sweep — run when tuning chunk_size/overlap/top_k
./.venv/bin/python3 evals/sweep.py

# Answer-quality eval — run on demand, pick a provider
./.venv/bin/python3 evals/run_answer_eval.py --provider ollama
./.venv/bin/python3 evals/run_answer_eval.py --provider anthropic

# pytest gate — run before committing / in CI
./.venv/bin/python3 -m pytest evals/test_regressions.py -v -s
```

## Reading the output

- **hit@k**: did at least one of a case's `expected_sources` files appear in the retrieved/relevant set? Coarse and forgiving — satisfied even if the right doc is the 3rd of 3 results.
- **precision@k**: fraction of retrieved chunks that came from `expected_sources`. **Can look mediocre even when the answer would be entirely correct** — e.g. a notice-period question might legitimately pull in three different documents that all happen to mention notice periods, but only one was marked "expected" for that specific case. Low precision means noisy retrieval, not necessarily a wrong final answer.
- **recall**: fraction of a case's `expected_sources` that were actually represented at all. Unlike precision, this one matters on its own — if a required source never shows up, the LLM structurally could not have used it.
- **leak count**: chunks (`run_retrieval_eval.py`) or fingerprint strings (`run_answer_eval.py`) from `forbidden_sources` that showed up where they shouldn't have. **Zero tolerance** — any non-zero leak in a non-`known_limitation` case is a correctness bug, not a tuning tradeoff, and `test_regressions.py` fails the build on it.
- **guardrail accuracy**: did `apply_relevance_guardrail()` empty the result exactly when `expect_guardrail` said it should? Note the two `other_personal_doc` cases in the dataset are deliberately asymmetric — one is blocked by the guardrail, the other by access control one layer earlier — see the comment above them in `dataset.py`.
- **`known_limitation` cases**: reported completely separately in every script's output, and excluded from every pass/fail gate in `test_regressions.py`. They represent a real, accepted, tracked gap (currently: the guardrail's `EMP\d+` regex doesn't catch a name used instead of an ID) — the point is to *see* them fail on every run, not to silently patch the dataset until they stop.

## A caveat about `run_answer_eval.py`'s checks

Neither of its two heuristics is a substitute for reading the actual answers:

- `looks_like_no_answer()` is a plain keyword list (`"not mentioned"`, `"no information"`, ...). It has already produced a real false negative in practice — an answer saying *"the context does **not mention**..."* was missed because the list has `"not mentioned"` (different tense). It detects refusal-shaped text, nothing about whether a non-refusal answer is actually correct.
- The content-leak check only looks at the final answer text. A case can show a real, reproducible leak at the retrieval level (`run_retrieval_eval.py`, deterministic) while the same case's answer-level check comes back clean on a given run, simply because the LLM happened not to take the bait that time. Don't treat a clean `run_answer_eval.py` result as proof a `known_limitation` case is fixed — check `run_retrieval_eval.py`'s leak count for that, since it doesn't depend on LLM non-determinism.
