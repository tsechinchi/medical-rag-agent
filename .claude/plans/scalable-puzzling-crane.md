# Implementation Plan: Metrics Adjustment & System Modifications for Direct Evidence-Based Answers

## Context

The system is currently designed as **safety-first**, which results in:
- **0.0 scores** for RAGAS metrics (answer_relevancy, context_precision) when system abstains from answering
- **1.0 scores** for NLI faithfulness when system correctly abstains
- **Appropriate behavior** (prioritizing safety over speculation), but evaluation metrics don't reflect this design goal

The user is requesting:
1. **Adjust evaluation metrics** to better reflect a safety-first design (not penalize correct abstentions)
2. **Modify the system** to provide more direct evidence-based answers (lower thresholds, enable partial answers)

---

## Root Cause Analysis

### Current Safety Mechanisms (Why System Abstains)
- **Evidence floor gate:** `LOW_EVIDENCE_SCORE_FLOOR = 0.2` (skips generation if top doc score below 20%)
- **NLI critic validation:** Requires `FAITHFULNESS_THRESHOLD = 0.7` (70%+ entailment)
- **Retry mechanism:** `MAX_RETRIES = 2` (gives up after 2 failed attempts)
- **Per-sentence support:** `CRITIC_SENTENCE_SUPPORT_THRESHOLD = 0.7` (each sentence must be 70%+ supported)

### Why Metrics Show 0s
- When system abstains (returns "The available evidence does not directly address this question")
- RAGAS metrics require an actual answer to evaluate (they can't score abstention)
- Result: Shows 0.0 instead of NULL/NA, appearing as "failed" rather than "strategically withheld"

### Missing Visibility
- Internal safety metrics exist in code but aren't exposed to evaluation CSV:
  - `unsupported_claims_count` (per-query)
  - `faithfulness_score` (NLI score from critic)
  - `retry_count` (how many retries occurred)
  - `critic_feedback` (explanation of validation failures)

---

## Three-Phase Implementation Strategy

### PHASE 0: Metrics Adjustment (Priority: IMMEDIATE - Must Do First)

**Why First:** Enables safe monitoring of downstream changes.

**Changes:**

#### 1. Expose Safety Metrics in CSV
File: `/teamspace/studios/this_studio/src/evaluation/model_free_eval.py` (lines 191-205)

Add new columns to DataFrame before writing CSV:
- `abstention_detected`: 1 if answer equals the "available evidence" boilerplate, 0 otherwise
- `unsupported_claims_count`: Number of claims rejected by critic NLI validation
- `citation_count`: Number of [1], [2], [3] citations in final answer
- `retry_count`: How many retry attempts were needed for this query
- `evidence_score`: Confidence score of top retrieved document

#### 2. Mask RAGAS Metrics for Abstention Rows
File: `/teamspace/studios/this_studio/src/evaluation/merge_results.py` (lines 59-84)

When merging results:
- For rows where `abstention_detected == 1`: set `answer_relevancy`, `context_precision`, `context_recall` to NULL (not 0.0)
- These metrics don't apply to abstentions and shouldn't skew averages

#### 3. Add Abstention Precision Metric
File: `/teamspace/studios/this_studio/src/evaluation/merge_results.py`

New metric: `abstention_precision = (# of correct abstentions) / (# total abstentions)`
- Correct abstention = faithfulness_nli == 1.0 AND abstention_detected == 1
- This measures whether the system is abstaining appropriately

**Expected Outcome:**
- CSV now has columns: `abstention_detected`, `unsupported_claims_count`, `citation_count`, `retry_count`, `evidence_score`
- RAGAS metrics for abstention rows show NULL instead of 0.0
- Summary includes `abstention_precision` metric


### PHASE 1: Safe Threshold Adjustments (Priority: HIGH - After Phase 0)

**Why Second:** Low-risk changes that gradually enable more answers without sacrificing safety.

**Changes:**

#### 1. Configuration Changes
File: `/teamspace/studios/this_studio/config/config.py`

```python
# Current → Proposed
MAX_RETRIES = 2                                 # → 3  (one extra attempt)
LOW_EVIDENCE_SCORE_FLOOR = 0.2                 # → 0.25  (5% relaxation)
# Do NOT change FAITHFULNESS_THRESHOLD yet - wait for Phase 2
```

#### 2. Add Safety Anchor to Prompts
File: `/teamspace/studios/this_studio/src/model/prompts.py` (after line 19)

Insert safety reminder that reinforces evidence-grounding without restricting answers:

```python
SAFETY_ANCHOR = (
    "\n\nCRITICAL: Every claim must be directly traceable to the provided context. "
    "If uncertain about supporting evidence, state the limitation: "
    "'The context suggests X, but does not clearly establish Y.' "
    "Abstention is better than speculation."
)
```

Append this to `SYSTEM_PROMPT` (line 19).

**Risk Mitigation:**
- MAX_RETRIES: 2→3 adds ~0.5-1s latency per query (test in evaluation)
- LOW_EVIDENCE_SCORE_FLOOR: 0.2→0.25 allows 5% more answers, but still conservative
- Safety anchor reinforces evidence requirement without changing behavior

**Validation:**
- Run evaluation, compare `unsupported_claims_count` before/after
- Check if `abstention_precision` remains >= 0.9 (90%+ of abstentions are appropriate)
- Monitor latency metrics


### PHASE 2: Critic Relaxation & Partial Answers (Priority: MEDIUM - After Phase 1 Validation)

**Only after Phase 1 shows no safety degradation.**

**Changes:**

#### 1. Relax Critic Thresholds
File: `/teamspace/studios/this_studio/config/config.py`

```python
FAITHFULNESS_THRESHOLD = 0.7                   # → 0.65  (5% relaxation)
CRITIC_SENTENCE_SUPPORT_THRESHOLD = 0.7       # → 0.65  (per-sentence)
```

#### 2. Enable Partial Answer Generation
File: `/teamspace/studios/this_studio/src/graph/nodes/generator.py` (lines 62-77)

Modify logic to support three answer paths instead of two (abstain vs. generate):

```python
# Current: if top_score < 0.2: abstain
# New:
if top_score < 0.1:
    # Completely absent - abstain
    draft_answer = "The available evidence does not directly address this question."
elif 0.1 <= top_score < 0.25:
    # Weak signal - generate with confidence marker
    # Generate answer normally, but prepend "[Partially Supported]"
    draft_answer = "[Partially Supported] " + generated_answer
else:
    # Normal generation path
    draft_answer = generated_answer
```

#### 3. Add Confidence Instructions to Prompts
File: `/teamspace/studios/this_studio/src/model/prompts.py` (lines 47-73, in mode_instructions)

Add to all mode instructions:

```python
"If the context supports only PART of your answer, be explicit about limitations: "
"'The evidence supports X, but the context does not establish Y.' "
"When you prepend [Partially Supported], it means the answer is grounded but incomplete."
```

**Risk Mitigation:**
- FAITHFULNESS_THRESHOLD reduction pairs with CRITIC_SENTENCE_SUPPORT_THRESHOLD reduction
- Partial answer markers make confidence level explicit
- Monitor `unsupported_claims_count` for degradation
- If unsupported claims spike, revert CRITIC_SENTENCE_SUPPORT_THRESHOLD immediately

**Validation:**
- Compare `unsupported_claims_count` and `citation_count` metrics before/after
- Verify partial answers are tagged correctly
- Check user satisfaction with confidence-labeled responses


### PHASE 3: Confidence Display in Responses (Optional - After Phase 2)

File: `/teamspace/studios/this_studio/src/graph/nodes/synthesizer.py` (lines 94-100)

Add optional feature to append confidence label to final answer:

```python
CONFIDENCE_LEVEL_MAP = {
    (0.9, 1.0]: "Strongly Supported",
    (0.7, 0.9]: "Well-Supported",
    (0.5, 0.7]: "Partially Supported",
    (0.0, 0.5]: "Weakly Supported",
}

if config.INCLUDE_CONFIDENCE_LABELS:
    confidence_label = CONFIDENCE_LEVEL_MAP[faithfulness_score]
    final_answer += f"\n[Evidence Confidence: {confidence_label}]"
```

---

## Critical Files to Modify

| File | Phase | Lines | Change Type |
|------|-------|-------|------------|
| config/config.py | 1, 2 | 39, 65, 38, 43 | Threshold values |
| src/evaluation/model_free_eval.py | 0 | 191-205 | Add columns |
| src/evaluation/merge_results.py | 0 | 59-84, new | Mask RAGAS, add abstention_precision |
| src/model/prompts.py | 1, 2 | 19, 47-73 | Add safety anchor, confidence instructions |
| src/graph/nodes/generator.py | 2 | 62-77 | Ternary logic for partial answers |
| src/graph/nodes/synthesizer.py | 3 | 94-100 | Confidence display (optional) |

---

## Verification & Testing Strategy

### Phase 0 Validation
1. Run evaluation: `python -m src.evaluation.run_eval`
2. Check output CSV (`experiments/model_free_eval_results.csv`) has new columns
3. Verify `abstention_detected` correctly identifies boilerplate answers
4. Verify RAGAS metrics show NULL for abstention rows in `all_results.csv`

### Phase 1 Validation
1. Run evaluation with new config
2. Compare `unsupported_claims_count` vs baseline (should not increase)
3. Check `abstention_precision` (should be >= 0.9)
4. Verify `latency_per_query_s` (acceptable if < 2x baseline)

### Phase 2 Validation
1. Run evaluation with relaxed thresholds
2. Count rows with `[Partially Supported]` tag (should be 5-15% of non-abstentions)
3. Verify `unsupported_claims_count` does not increase significantly
4. Check citation accuracy with new `citation_count` metric

### Phase 3 Validation (if implemented)
1. Verify confidence labels appear in final answers
2. Spot-check that labels match `faithfulness_score` values
3. User testing: clarity of confidence levels

---

## Recommended Implementation Order

1. **Week 1:** Implement Phase 0 (metrics exposure)
   - Run evaluation to establish baseline with new metrics

2. **Week 2:** Implement Phase 1 (safe threshold changes + safety anchors)
   - Run evaluation, compare against Phase 0 baseline
   - If unsupported_claims_count stable, proceed

3. **Week 3:** Implement Phase 2 (critic relaxation + partial answers)
   - Run evaluation, monitor safety
   - If safety remains high, prepare Phase 3

4. **Week 4:** (Optional) Implement Phase 3 (confidence display)

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|-----------|
| MAX_RETRIES 2→3 | LOW | Monitor latency, set timeout limit |
| LOW_EVIDENCE_SCORE_FLOOR 0.2→0.25 | LOW | Small 5% change, conservative threshold pairing |
| FAITHFULNESS_THRESHOLD 0.7→0.65 | MEDIUM | Requires strong critic sentence validation; monitor unsupported_claims |
| CRITIC_SENTENCE_SUPPORT_THRESHOLD 0.7→0.65 | MEDIUM-HIGH | Only implement with SAFETY_ANCHOR in prompts; immediate revert if claims spike |
| Partial answer markers | LOW | Explicit labeling makes confidence clear; easy to toggle off |

---

## Success Criteria

✅ **Phase 0:** All new safety metrics appear in CSV, RAGAS metrics NULL for abstentions
✅ **Phase 1:** Unsupported claims don't increase, abstention_precision >= 0.9, latency acceptable
✅ **Phase 2:** Partial answers are appropriately tagged (5-15% of answers), safety maintained
✅ **Phase 3:** Confidence labels clearly communicate evidence strength

---

## User Preferences (Confirmed)

✅ **Test Set Size:** Start with current 3-question test set for rapid iteration and stability validation, then expand
✅ **Confidence Display:** Include confidence markers in final_answer (user-facing)
✅ **Partial Answer Thresholds:** Use moderate approach:
   - 0.0-0.10: Full abstention ("available evidence")
   - 0.10-0.25: Partial answer with confidence marker ("[Partially Supported]")
   - 0.25+: Normal full answer generation
✅ **Validation Approach:** Sequential validation after each phase before proceeding to next

---

## Implementation Details (Finalized)

### Partial Answer Logic (Updated)
File: `/teamspace/studios/this_studio/src/graph/nodes/generator.py` (lines 62-77)

```python
# evidence_score is the rerank score of top document
if evidence_score < 0.10:
    # No meaningful evidence - abstain
    draft_answer = "The available evidence does not directly address this question."
    confidence_level = 0.0
elif 0.10 <= evidence_score < 0.25:
    # Weak signal - attempt generation with partial confidence marker
    draft_answer = generated_answer_from_llm(context, prompt)
    draft_answer = "[Partially Supported] " + draft_answer
    confidence_level = 0.5
else:
    # Normal generation path
    draft_answer = generated_answer_from_llm(context, prompt)
    confidence_level = max(nli_faithfulness_score, 0.7)
```

### Confidence Display in Final Answer
File: `/teamspace/studios/this_studio/src/graph/nodes/synthesizer.py` (lines 94-100)

Add after line 95 (answer cleaning):

```python
# Map evidence scores to human-readable confidence labels
CONFIDENCE_LABELS = {
    (0.9, 1.0]: "Strongly Supported by Evidence",
    (0.7, 0.9]: "Well-Supported by Evidence",
    (0.5, 0.7]: "Partially Supported; Context Limitations Noted",
    (0.0, 0.5]: "Weakly Supported; Treat As Provisional",
}

# Add confidence label to final answer
if state.get("confidence_level", 0) > 0:
    for (low, high), label in CONFIDENCE_LABELS.items():
        if low < state["confidence_level"] <= high:
            final_answer += f"\n[{label}]"
            break
```

### Phase 1 Config Changes (Refined)
File: `/teamspace/studios/this_studio/config/config.py`

```python
# From Phase 1
MAX_RETRIES = 2                          # → 3
LOW_EVIDENCE_SCORE_FLOOR = 0.2           # → 0.10 (matches user's 0.10 threshold)
# From Phase 2 (sequential validation)
FAITHFULNESS_THRESHOLD = 0.7             # → 0.65
CRITIC_SENTENCE_SUPPORT_THRESHOLD = 0.7  # → 0.65
```

---

## Updated Validation Timeline

### Phase 0: Metrics (Day 1)
1. Add safety columns to CSV
2. Run eval on 3-question set
3. Verify new metrics appear correctly

### Phase 1: Thresholds (Day 2, after Phase 0 validation)
1. Update config: MAX_RETRIES, LOW_EVIDENCE_SCORE_FLOOR
2. Run eval on same 3-question set
3. Check: unsupported_claims_count, abstention_precision, latency

### Phase 2: Critic Relaxation (Day 3, after Phase 1 validation)
1. Update config: FAITHFULNESS_THRESHOLD, CRITIC_SENTENCE_SUPPORT_THRESHOLD
2. Run eval on same 3-question set
3. Safety checks, then expand test set

### Expansion Decision (After Phase 2 stability)
- If safety metrics look good, expand to 20-50 question test set
- Rerun all phases with larger set to confirm stability at scale
