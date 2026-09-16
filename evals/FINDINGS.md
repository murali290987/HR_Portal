# Eval harness findings

A record of what `evals/sweep.py` and manual investigation actually
found — so the repo can answer "did the sweep run, what won, and why"
without re-running it. Regenerate the sweep table with:

```bash
./.venv/bin/python3 evals/sweep.py
```

## Winning config: the current defaults, unchanged

`config.py`'s existing defaults — `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`,
`TOP_K=3` — are already in the top tier after the fixes below. No
config change was made as a result of this sweep.

```
chunk_size  overlap  top_k  hit@k   leaks  known_leaks  guardrail_ok
---------------------------------------------------------------------
300         0        3       100%   0      2            21/21
300         0        5       100%   0      2            20/21
300         50       5       100%   0      2            20/21
300         150      5       100%   0      2            20/21
500         0        5       100%   0      2            20/21
500         50       3       100%   0      2            21/21   <- current defaults
500         50       5       100%   0      2            20/21
500         150      3       100%   0      2            21/21
500         150      5       100%   0      2            20/21
800         0        3       100%   0      2            20/21
...
1200        50       3       100%   0      2            21/21
...
300         50       3        93%   0      2            21/21
300         150      3        93%   0      2            21/21
500         0        3        93%   0      2            21/21
1200        0        1        87%   0      1            21/21
...
500         150      1        60%   0      1            21/21
```

(Full 36-row table reproducible via `evals/sweep.py`; truncated here to
the rows that matter for the decision.)

**Read on `TOP_K`:** `TOP_K=1` is unambiguously worse (60–87% hit@k)
than `TOP_K=3` or `TOP_K=5` (93–100%, mostly a clean 100%) across every
chunk size/overlap combination. This is a real, measured signal, not a
guess.

**Read on chunk size/overlap:** at `TOP_K≥3`, nearly every
`(chunk_size, overlap)` combination reaches 100% hit@k — this small a
corpus doesn't meaningfully discriminate between them on hit@k alone.
`guardrail_ok` varies between `20/21` and `21/21` depending on config
(see "Known limitation: guardrail unreliability at higher TOP_K"
below) — the current defaults are one of the configs that hit `21/21`.

**Note on this table's history:** an earlier version of this sweep was
run before the chunker tail-merge fix (see below) and is not
reproduced here — those numbers were measured against a corpus that
included a near-content-free duplicate chunk on some configs. Only
this post-fix run should be treated as reliable.

## Negative result: a blanket embedding-distance cutoff (abandoned)

Before the current relevance guardrail (`apply_relevance_guardrail()`,
matching an explicit `EMP\d+` pattern), a blanket cutoff on embedding
distance was tried as a general "is this even relevant" filter. It was
abandoned and never shipped: with a corpus this small and a
lightweight embedding model (`all-MiniLM-L6-v2`), genuinely relevant
and genuinely irrelevant matches land in *overlapping* distance
ranges. Concretely: a real answer to "What is my CTC?" scored a worse
(higher) distance than a genuinely irrelevant match for an unrelated
question. No single threshold could separate them without either
missing real bugs or blocking real answers. See `config.py`'s comment
above `EMPLOYEE_ID_PATTERN` and `ARCHITECTURE.md` §7 for the full
writeup.

## Fixed: chunker tail-merge bug

`ingest.py`'s `chunk_text()` could produce a final chunk containing
almost no new content — confirmed case: `chunk_size=500, overlap=50`
on `05_leave_policy.md` produced a 59-character final chunk, 50 of
which were pure duplicate overlap text (9 characters actually new).
Mathematically the tail can never be *shorter than* `overlap` with
this sliding-window algorithm (always `overlap < tail ≤ chunk_size`),
so the fix merges a tail chunk when it contributes *less new content
than one overlap-window* (`tail_len < 2 × overlap`) into the previous
chunk instead. Verified zero near-duplicate tails remain across the
full 36-config grid after the fix. The real production
`faiss_index.bin` was affected by this (default config is exactly
`chunk_size=500, overlap=50`) and has been rebuilt.

## Fixed: HR role first-person bypass

`HR001` asking a vague first-person question ("what is my name") used
to return `"Priya Ramanathan."` — the `hr` role's ownership bypass in
`retrieve()` made *any* personal chunk eligible regardless of owner,
and the relevance guardrail's `EMP\d+` regex never fires on a bare
pronoun. Fixed in `retrieve()` itself: the hr bypass now only admits a
personal chunk when the query names its specific owner, by id or by
name (`_query_names_employee()`, `config.EMPLOYEE_NAMES`). Verified
against the original bug and both existing legitimate hr-role cases
(asking by id, asking by name) — no regression. Tracked going forward
as `hr_first_person_no_subject_blocked`, a real (not known-limitation)
gate.

## Investigated, not confirmed: table-shredding (item 6)

The hypothesis that fixed-size chunking shreds the per-diem/hotel-cap
table in `06_expense_travel_policy.md` was checked directly against
the actual chunk boundaries under the current config — **it does
not**. All three grade rows (`Trainee/Associate`, `Senior
Associate/Manager`, `Senior Manager & above`) are fully contained
within a single chunk. `table_lookup`'s eval numbers also aren't
uniquely worst — it ties with `hr_role` and `own_personal_doc` at 33%
precision@k, with hit@k a perfect 100% across all three.

The earlier live finding that the model answered a "Senior Manager"
per-diem question with the "Senior Associate/Manager" row's value
(`INR 2,000` instead of `2,500`) is real, but it's a **generation-side
reading error over an intact table**, not a retrieval or chunking
symptom — structure-aware chunking would not have prevented it.
**Decision: not pursuing structure-aware/heading-based chunking based
on this evidence.**

## Confirmed, left open: appraisal-letter chunking gap (item 7)

Real and reproducible: `04_appraisal_letter_sample.md`'s header chunk
(name/designation/department) and its revised-compensation table land
in *separate* chunks, and the header outscores the table chunk for
queries like "What is my revised CTC after the appraisal cycle?" — the
chunk with the actual number (`16,90,000`) misses the top-3 entirely.

**This gap is invisible to `hit@k` as currently defined.** `hit@k` is
file-level ("did `04_appraisal_letter_sample.md` show up anywhere"),
and it does — via the header chunk — so `hr_full_appraisal_lookup`
reads `hit_at_k: True` even though the chunk that actually answers the
question is missing. Only Stage 4's `expected_answer_contains` check
(`["16,90,000", "senior software engineer"]`) catches this, and Stage
4 (real LLM calls) is not part of the `test_regressions.py` pytest
gate. **Left unfixed for now** (same decision as item 6 — not pursuing
structure-aware chunking on current evidence), but worth revisiting if
more documents in this shape get added.

## Leak severity split (access_violation vs. wrong_subject)

`evals/dataset.py` cases with a real leak now carry a `leak_type`:

- **`access_violation`** — the requester has no legitimate relationship
  to the data at all (e.g. `other_id_named_as_stranger`: `EMP99999`
  asking about `EMP10453`'s CTC). Access control alone should block
  it, and does.
- **`wrong_subject`** — the requester has a legitimate way to see this
  exact data (it's genuinely theirs, or their role legitimately grants
  broad access), but the question is about someone else
  (`other_id_named_as_owner`, `known_limitation_name_bypass`,
  `hr_first_person_no_subject_blocked`). A subtler bug class: the
  permission check was correct, the failure is entirely about subject
  misattribution.

Both are zero-tolerance in `test_regressions.py` — the split changes
reporting/triage, not which leaks are allowed to pass.

## Known limitation: guardrail unreliability at higher `TOP_K`

At higher `TOP_K`, a stray general chunk can survive alongside the
guardrail's filtering, so `guardrail_triggered` reads `False` even
though no personal data leaked — no security issue, but the
deterministic "never call the LLM" block becomes less reliable,
silently falling back to the LLM's own judgment instead. Visible in
the sweep table above as `guardrail_ok` dropping from `21/21` to
`20/21` at some `(chunk_size, overlap, TOP_K)` combinations.
