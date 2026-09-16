"""
The golden dataset for the eval harness — ~20 hand-built test cases,
grounded in what's actually written in hr_docs/ (verified by reading
every file, not guessed).

Why a .py file instead of .json: a few cases need a short comment
explaining a non-obvious expectation (e.g. why one case expects
guardrail_triggered=False even though it looks like it should block
something), and JSON has no comment syntax. Since nothing here needs to
be read by a non-Python tool, the tradeoff is one-sided — plain Python
literals also mean no separate parsing/validation step, and a case with
a typo'd field name fails immediately and loudly (KeyError) rather than
silently producing an empty value the way a JSON typo might.

Field meanings:
    id                       short stable slug for reporting
    category                 one of CATEGORIES below
    question                 the query text
    user_id                  key into config.USERS -- who's asking
    expected_sources         source_file names that SHOULD appear in the
                             retrieved set (by filename, not chunk_id --
                             retrieval operates on chunks, but "did the
                             right DOCUMENT show up" is the meaningful
                             check here)
    forbidden_sources        source_file names that must NEVER appear in
                             the retrieved set for this user -- a
                             non-zero hit here is a leak, not a quality
                             issue
    expect_guardrail         whether apply_relevance_guardrail() should
                             empty the result entirely (server.py's
                             guardrail_triggered). Cases where this is
                             True are skipped by Stage 4 (run_answer_eval.py)
                             entirely -- no LLM call happens for them in
                             production either, so there's no generated
                             answer to check.
    known_limitation         True for cases we already know FAIL today.
                             These are measured, not fixed -- see the
                             comment above case group 4.
    expected_answer_contains Stage 4 only: substrings (case-insensitive)
                             the final LLM answer should contain. Empty
                             for no_match cases (checked differently --
                             see looks_like_no_answer() in
                             run_answer_eval.py) and for the
                             expect_guardrail=True case (skipped
                             entirely).
    notes                    why the expected values are what they are
"""

# Category `hr_role` is an addition beyond the spec's 6-value enum --
# needed to distinguish "blocked by access control" (other_personal_doc)
# from "allowed via the RBAC bypass" (hr_role), which are different
# mechanisms and should be measured separately.
CATEGORIES = {
    "general_policy",
    "own_personal_doc",
    "other_personal_doc",
    "hr_role",
    "no_match",
    "multi_doc",
    "table_lookup",
}

CASES = [
    # ---- general_policy: straightforward retrieval, no access issues ----
    {
        "id": "gp_casual_leave",
        "category": "general_policy",
        "question": "How many casual leaves am I entitled to per year?",
        "user_id": "EMP10453",
        "expected_sources": ["01_hr_faq.md", "05_leave_policy.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["12"],
        "notes": "Both docs independently state 12 CL/year.",
    },
    {
        "id": "gp_wfh_days",
        "category": "general_policy",
        "question": "How many WFH days am I allowed per month?",
        "user_id": "EMP99999",
        "expected_sources": ["01_hr_faq.md", "07_wfh_hybrid_policy.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["8"],
        "notes": "Both docs state up to 8 WFH days/month.",
    },
    {
        "id": "gp_public_holidays",
        "category": "general_policy",
        "question": "How many public holidays does the company observe each year?",
        "user_id": "EMP10453",
        "expected_sources": ["05_leave_policy.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["10"],
        "notes": "'10 fixed public holidays per year' -- only in 05.",
    },
    {
        "id": "gp_dress_code",
        "category": "general_policy",
        "question": "What is the dress code policy?",
        "user_id": "EMP99999",
        "expected_sources": ["02_company_norms.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["business casual"],
        "notes": "Dress Code section only exists in 02.",
    },
    {
        "id": "gp_expense_claim_window",
        "category": "general_policy",
        "question": "Within how many days must I submit an expense claim after travel?",
        "user_id": "EMP10453",
        "expected_sources": ["02_company_norms.md", "06_expense_travel_policy.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["15"],
        "notes": "Both independently say 15 days.",
    },
    # ---- own_personal_doc: the requester's own confidential data ----
    {
        "id": "own_revised_ctc",
        "category": "own_personal_doc",
        "question": "What is my revised CTC after the appraisal cycle?",
        "user_id": "EMP10453",
        "expected_sources": ["04_appraisal_letter_sample.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["16,90,000"],
        "notes": "Revised CTC (16,90,000) only appears in the appraisal letter.",
    },
    {
        "id": "own_notice_period",
        "category": "own_personal_doc",
        "question": "What is my notice period after confirmation, per my offer letter?",
        "user_id": "EMP10453",
        "expected_sources": ["03_offer_letter_sample.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["60"],
        "notes": "60 days, stated in the offer letter's terms table.",
    },
    {
        "id": "own_appraisal_rating",
        "category": "own_personal_doc",
        "question": "What was my overall performance rating in the last appraisal cycle?",
        "user_id": "EMP10453",
        "expected_sources": ["04_appraisal_letter_sample.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["4.2"],
        "notes": "4.2 / 'Exceeds Expectations', only in the appraisal letter.",
    },
    # ---- other_personal_doc: a DIFFERENT employee's confidential data,
    # named explicitly enough for the guardrail (or access control) to
    # catch. Note the asymmetry between these two cases -- which layer
    # actually stops each one differs, and that's worth measuring:
    #   - id_named_as_owner: the requester genuinely owns a personal
    #     chunk, but names a DIFFERENT id -- access control alone would
    #     let it through (it's their own doc); the GUARDRAIL is what
    #     strips it. expect_guardrail=True.
    #   - id_named_as_stranger: the requester owns nothing personal at
    #     all, so access control strips Priya's docs before the
    #     guardrail ever runs -- the guardrail has nothing left to act
    #     on, so guardrail_triggered stays False even though the leak
    #     is (correctly) prevented. This is the same behavior confirmed
    #     live earlier in the session.
    {
        "id": "other_id_named_as_owner",
        "category": "other_personal_doc",
        "question": "Can you share the EMP99999 CTC?",
        "user_id": "EMP10453",
        "expected_sources": [],
        "forbidden_sources": ["03_offer_letter_sample.md", "04_appraisal_letter_sample.md"],
        "expect_guardrail": True,
        "known_limitation": False,
        "expected_answer_contains": [],
        "notes": (
            "EMP10453's own docs would otherwise be retrieved (they're genuinely "
            "hers); the guardrail strips them because the question names a "
            "different EMP id. This is the original reported bug, now fixed. "
            "expect_guardrail=True means Stage 4 skips this case entirely -- "
            "the fixed message is returned with no LLM call, same as production."
        ),
    },
    {
        "id": "other_id_named_as_stranger",
        "category": "other_personal_doc",
        "question": "What is EMP10453's CTC?",
        "user_id": "EMP99999",
        "expected_sources": [],
        "forbidden_sources": ["03_offer_letter_sample.md", "04_appraisal_letter_sample.md"],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": [],
        "notes": (
            "EMP99999 owns no personal docs, so access control strips Priya's "
            "chunks before the guardrail runs. guardrail_triggered is correctly "
            "False here -- the block happened one layer earlier. No positive "
            "fact to expect in the answer; Stage 4 relies on the content-leak "
            "check (must not contain a 03/04 fingerprint) rather than a refusal-"
            "phrasing check, since this isn't a no_match case by category."
        ),
    },
    # ---- other_personal_doc, known limitation: the guardrail's regex
    # only matches EMP\d+ patterns. A question that implies a DIFFERENT
    # identity via a NAME (not an id) is invisible to it. The requester
    # here owns real personal data (access control correctly allows it),
    # but the question names someone else -- and nothing catches that.
    # This is the same bug class as other_id_named_as_owner above,
    # reproduced via name instead of id. NOT to be fixed as part of this
    # eval harness -- see the top-level prompt.
    {
        "id": "known_limitation_name_bypass",
        "category": "other_personal_doc",
        "question": "What is Rahul Mehta's CTC?",
        "user_id": "EMP10453",
        "expected_sources": [],
        "forbidden_sources": ["03_offer_letter_sample.md", "04_appraisal_letter_sample.md"],
        "expect_guardrail": False,
        "known_limitation": True,
        "expected_answer_contains": [],
        "notes": (
            "No employee named Rahul Mehta exists in hr_docs. EMP10453's own "
            "CTC chunk will be retrieved (access control correctly allows it -- "
            "it's genuinely hers) and the guardrail does nothing, since its "
            "regex only matches EMP\\d+ patterns, not names. Expected to FAIL "
            "(forbidden_sources leak > 0 at retrieval, and likely a content-level "
            "leak in Stage 4's answer too) until the guardrail is extended to "
            "reason about named entities, not just explicit ids."
        ),
    },
    # ---- hr_role: the RBAC bypass, succeeding where an employee role fails ----
    {
        "id": "hr_full_appraisal_lookup",
        "category": "hr_role",
        "question": "What is EMP10453's revised CTC and revised designation after appraisal?",
        "user_id": "HR001",
        "expected_sources": ["04_appraisal_letter_sample.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["16,90,000", "senior software engineer"],
        "notes": "HR role bypasses ownership check entirely; id is named and matches, so nothing blocks it.",
    },
    {
        "id": "hr_offer_letter_lookup",
        "category": "hr_role",
        "question": "What is Priya Ramanathan's designation and department, according to her offer letter?",
        "user_id": "HR001",
        "expected_sources": ["03_offer_letter_sample.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["software engineer ii", "platform engineering"],
        "notes": "Same document an EMP99999 employee-role request would be blocked from.",
    },
    # ---- hr_role, known limitation: a SECOND, distinct bypass from
    # known_limitation_name_bypass above. That one required the asker to
    # name a specific wrong third party. This one needs no name at all --
    # just a vague first-person question ("my ...") asked under a role
    # (hr) that bypasses ownership entirely. HR001 owns no personal
    # document of its own, so there's no legitimate referent for "my" --
    # but retrieve() doesn't know that; it just returns the closest
    # matching personal chunk regardless of whose it is, and
    # apply_relevance_guardrail() only checks for an explicit EMP\d+
    # pattern, which a first-person pronoun never contains. Discovered
    # live: HR001 asking "what is my name" got back "Priya Ramanathan."
    {
        "id": "known_limitation_hr_first_person_bypass",
        "category": "hr_role",
        "question": "What is my name?",
        "user_id": "HR001",
        "expected_sources": [],
        "forbidden_sources": ["03_offer_letter_sample.md", "04_appraisal_letter_sample.md"],
        "expect_guardrail": False,
        "known_limitation": True,
        "expected_answer_contains": [],
        "notes": (
            "HR001 has no personal document of its own -- 'my name' has no "
            "legitimate referent. The HR role's ownership bypass makes ANY "
            "personal chunk eligible regardless of who it belongs to, and the "
            "guardrail's EMP\\d+ regex never fires on a bare pronoun with no ID "
            "or name in the query. Confirmed live: answered 'Priya Ramanathan.' "
            "Broader than known_limitation_name_bypass -- needs no specific "
            "wrong name, just a vague personal question under the hr role."
        ),
    },
    # ---- no_match: nothing in the corpus answers this ----
    {
        "id": "no_match_stock_options",
        "category": "no_match",
        "question": "What is the company's stock option vesting schedule?",
        "user_id": "EMP10453",
        "expected_sources": [],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": [],
        "notes": "No mention of stock options/ESOPs anywhere in hr_docs. Checked via looks_like_no_answer(), not expected_answer_contains.",
    },
    {
        "id": "no_match_referral_bonus",
        "category": "no_match",
        "question": "What is the employee referral bonus amount?",
        "user_id": "EMP99999",
        "expected_sources": [],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": [],
        "notes": "No referral program mentioned anywhere in hr_docs. Checked via looks_like_no_answer().",
    },
    # ---- table_lookup: markdown tables, likely to get shredded by
    # fixed-size chunking -- exactly what this eval is meant to catch ----
    {
        "id": "table_per_diem_senior_manager",
        "category": "table_lookup",
        "question": "What is the per diem for a Senior Manager on domestic travel?",
        "user_id": "EMP10453",
        "expected_sources": ["06_expense_travel_policy.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["2,500"],
        "notes": "Table row: Senior Manager & above -> INR 2,500 per diem.",
    },
    {
        "id": "table_hotel_cap_trainee",
        "category": "table_lookup",
        "question": "What is the hotel cap for a Trainee or Associate on domestic travel?",
        "user_id": "EMP99999",
        "expected_sources": ["06_expense_travel_policy.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["4,000"],
        "notes": "Table row: Trainee/Associate -> INR 4,000 hotel cap/night.",
    },
    {
        "id": "table_air_travel_class",
        "category": "table_lookup",
        "question": "What class of air travel is allowed for Senior Associate/Manager grade?",
        "user_id": "EMP10453",
        "expected_sources": ["06_expense_travel_policy.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["economy"],
        "notes": "Table row: Senior Associate/Manager -> Economy (flexible).",
    },
    # ---- multi_doc: the answer is scattered across more than one file ----
    {
        "id": "multi_notice_period",
        "category": "multi_doc",
        "question": "What is the notice period for resignation?",
        "user_id": "EMP10453",
        "expected_sources": ["01_hr_faq.md", "02_company_norms.md", "08_exit_process.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["60"],
        "notes": "All three independently state 30/60 days by grade.",
    },
    {
        "id": "multi_exit_process",
        "category": "multi_doc",
        "question": "What happens during the exit process, from resignation to final settlement?",
        "user_id": "EMP99999",
        "expected_sources": ["02_company_norms.md", "08_exit_process.md"],
        "forbidden_sources": [],
        "expect_guardrail": False,
        "known_limitation": False,
        "expected_answer_contains": ["45"],
        "notes": "02 has a brief Exit Process section; 08 has the full detailed process. 45 days = F&F settlement window.",
    },
]


def validate():
    """Fail loudly at import time if a case is missing a required field or uses a bad category."""
    required_fields = {
        "id", "category", "question", "user_id", "expected_sources",
        "forbidden_sources", "expect_guardrail", "known_limitation",
        "expected_answer_contains", "notes",
    }
    ids_seen = set()
    for case in CASES:
        missing = required_fields - case.keys()
        if missing:
            raise ValueError(f"Case {case.get('id', '?')} missing fields: {missing}")
        if case["category"] not in CATEGORIES:
            raise ValueError(f"Case {case['id']} has unknown category: {case['category']!r}")
        if case["id"] in ids_seen:
            raise ValueError(f"Duplicate case id: {case['id']}")
        ids_seen.add(case["id"])


validate()

if __name__ == "__main__":
    from collections import Counter

    print(f"{len(CASES)} cases loaded, validation passed.\n")
    print("By category:")
    for cat, count in Counter(c["category"] for c in CASES).items():
        print(f"  {cat}: {count}")
    known_limitations = [c["id"] for c in CASES if c["known_limitation"]]
    print(f"\nKnown-limitation cases (expected to fail): {known_limitations}")
