# Semantic decision pipeline refactor report

Date: 2026-08-18  
Semantic policy version: `v2`  
Blind holdout status: `LOCKED_DO_NOT_TUNE`  
Blind holdout SHA-256: `c9c4694f155ce8be829539f9fb03a62bcb82e9694605b2c00138daaa59a14648`

## Executive result

The backend semantic decision path is now policy-driven: semantic parsing produces a relation class, candidate generation stays separate, eight validators emit structured `GateResult` objects, the generic evaluator applies configured gate order, and `_candidate_rejection()` only returns the first failed gate reason.

On the same 260-row locked holdout, overall F1 increased from 25.3104 to 26.5639 and answerable F1 increased from 19.4578 to 20.9392. EM decreased slightly from 13.0769 to 12.6923. No-answer accuracy and false-positive rate were unchanged; false negatives increased from 91 to 97. This is a mixed result rather than a uniform quality win, but there is no strong aggregate semantic-quality regression.

## Required 24-point report

1. **Semantic `if/elif` in `backend/viqa_api.py` before refactor:** 31 decision-oriented branches in the pre-refactor audit.
2. **After refactor:** 2 generic/non-policy branches remain in the same audit: defaulting `relation_weight` in score composition and a subject-followed-by-relation formatting check. Neither decides a candidate by relation class. Candidate-specific `relation_type == ...` and `candidate.method == ...` branches are absent. Retriever selection branches (`bm25`, `hybrid`, `dense`) are unrelated to semantic candidate policy.
3. **Thresholds moved to config:** boundary 0.20, answer type 0.50, evidence 0.50, completeness 0.50, ranking 0.60; sentence fallback generation 0.42 and answer type 0.75; cause predicate/pattern/subject/target values; subject token overlap 0.75; boundary assessment 0.50; location coverage/quality/evidence/entity-relation values. Method penalties are also configured.
4. **Policies:** 15 relation policies: `GENERAL`, `GENERIC_LOCATION`, `CAUSE`, `BIRTH_TIME`, `DEATH_TIME`, `EVENT_TIME`, `EVENT_LOCATION`, `OBJECT_LOCATION`, `BIRTH_LOCATION`, `DEATH_LOCATION`, `PURPOSE`, `CONTRAST`, `DEFINITION`, `IDENTITY`, `ATTRIBUTE`. There are also three method policies: `neural_span`, `phrase_fallback`, `sentence_fallback`.
5. **Validators:** 8: Span, Boundary, Completeness, Evidence, Subject, Relation, AnswerType, Ranking.
6. **New `_candidate_rejection()` length:** 2 physical lines including the function declaration; its body is one return statement.
7. **Does rejection know CAUSE/LOCATION?** No. It reads `candidate.gate_results` through `first_gate_failure()` only.
8. **Duplicated sources of truth removed:** relation-specific rejection decisions left the backend; rejection messages/debug priorities moved to `reader/rejection_reasons.py`; relation dispatch is centralized in `reader/relation_validator.py`; thresholds/policies moved to `config/semantic_policy.json`; fallback selection moved to `reader/fallback_policy.py`; lexical normalization moved to `reader/lexical.py`; old backend relation/fallback branches were removed.
9. **Regression tests:** automated suite 119/119 passed. The existing 10-case live semantic regression artifact reports 10 fully correct, 0 partial, 0 wrong after the semantic changes.
10. **Blind holdout size:** 260 rows, stratified across CAUSE, TIME, LOCATION, PURPOSE, ENTITY, DEFINITION, GENERAL, including 40 unanswerable rows. It excludes the manual and known-regression sets and is protected by a hash lock.
11. **Blind holdout F1/EM:** final F1 26.5639; final EM 12.6923. Baseline F1 25.3104; baseline EM 13.0769.
12. **Answerable F1:** 20.9392 final versus 19.4578 baseline.
13. **No-answer accuracy:** 57.5% final versus 57.5% baseline.
14. **False-positive rate:** 42.5% (17/40) final versus 42.5% (17/40) baseline.
15. **False-negative rate:** 44.0909% (97/220) final versus 41.3636% (91/220) baseline.
16. **CAUSE:** overall F1 20.7796 versus 13.8736; answerable F1 13.8965 versus 5.9546.
17. **TIME:** overall F1 37.2883 versus 38.0291; answerable F1 39.4494 versus 40.2827.
18. **LOCATION:** overall F1 18.7831 versus 18.7831; answerable F1 2.5397 versus 2.5397.
19. **PURPOSE:** overall F1 33.1811 versus 39.8478; answerable F1 26.4764 versus 29.8097.
20. **Latency:** baseline average/p50/max = 4595.133/4478.120/8519.695 ms; final = 5370.990/4438.573/29606.635 ms. Median is effectively unchanged (-39.547 ms), while average and maximum were hurt by long-tail cases. The baseline cache was generated through localhost HTTP and final through in-process execution, so this is diagnostic rather than a strict transport-equivalent performance comparison.
21. **Rule coverage report:** `results/semantic_rule_coverage_v2.json`, 13 auditable rules evaluated on all 260 locked rows with matched/correct/incorrect/precision fields.
22. **LOW_COVERAGE rules:** 5 at the `< 5` match threshold: `CAUSE_NOMINAL_PREFIX` (3), `CAUSE_TRIGGER_PREFIX` (1), `CAUSE_MARKER_SUFFIX` (0), `TIME_BIRTH_MARKER` (0), `IDENTITY_INTERROGATIVE` (1). They are reported, not automatically removed or tuned. `TIME_DEATH_MARKER` is not low coverage (5 matches) but its 0.20 precision remains visible for future development-set review.
23. **Files changed:** see the grouped file inventory below.
24. **Benchmark proper names in production code/config:** none found by the final case-insensitive static search across `backend`, `reader`, and `config`. Production also does not import test/manual/regression fixtures.

## Before/after metrics

| Metric | Baseline | Refactored v2 final | Delta |
|---|---:|---:|---:|
| EM | 13.0769 | 12.6923 | -0.3846 |
| Overall F1 | 25.3104 | 26.5639 | +1.2535 |
| Answerable EM | 5.0000 | 4.5455 | -0.4545 |
| Answerable F1 | 19.4578 | 20.9392 | +1.4814 |
| No-answer accuracy | 57.5000% | 57.5000% | 0.0000 pp |
| False positives | 17 | 17 | 0 |
| False-positive rate | 42.5000% | 42.5000% | 0.0000 pp |
| False negatives | 91 | 97 | +6 |
| False-negative rate | 41.3636% | 44.0909% | +2.7273 pp |

| Relation | Baseline overall F1 | Final overall F1 | Baseline answerable F1 | Final answerable F1 |
|---|---:|---:|---:|---:|
| CAUSE | 13.8736 | 20.7796 | 5.9546 | 13.8965 |
| DEFINITION | 23.0769 | 26.6618 | 0.0000 | 4.6603 |
| ENTITY | 20.9411 | 26.9375 | 25.1293 | 32.3250 |
| GENERAL | 26.1946 | 24.0025 | 25.1238 | 20.1028 |
| LOCATION | 18.7831 | 18.7831 | 2.5397 | 2.5397 |
| PURPOSE | 39.8478 | 33.1811 | 29.8097 | 26.4764 |
| TIME | 38.0291 | 37.2883 | 40.2827 | 39.4494 |

## Ablation (final candidate pool)

| Variant | EM | F1 | Answerable F1 | No-answer accuracy | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Reader only | 3.4615 | 24.6299 | 25.0172 | 22.5% | 31 | 33 |
| + Answer refinement | 3.8462 | 22.0499 | 21.9681 | 22.5% | 31 | 38 |
| + Subject validation | 5.0000 | 22.5057 | 21.1431 | 27.5% | 29 | 56 |
| + Relation validation | 5.7692 | 20.5874 | 17.9670 | 35.0% | 26 | 70 |
| Full pipeline | 12.6923 | 26.5639 | 20.9392 | 57.5% | 17 | 97 |

The ablation exposes the trade-off instead of claiming every gate improves F1 independently: subject/relation gates reduce false positives but raise false negatives; completeness/fallback/final selection recover aggregate F1 in the full pipeline.

## Parser/generalization diagnostics

| Diagnostic | Baseline | Refactored |
|---|---:|---:|
| Fully parsed | 49.231% | 57.692% |
| Subject parsed | 53.846% | 63.462% |
| Predicate parsed | 82.308% | 87.308% |
| Relation parsed | 93.077% | 89.615% |
| UNKNOWN relation | 1.923% | 5.385% |
| Unseen paraphrase sanity | 3/12 (25.000%) | 11/12 (91.667%) |

Generalization sanity tests include the unseen synthetic rice-growth cause question and a synthetic birth-time entity. They are not part of the blind holdout.

## Gate trace integrity

The final cache contains 260 unique holdout IDs and 15,193 serialized candidates. Every candidate contains both keyed `gate_results` and ordered `gates`; no candidate is missing a structured trace. The cache metadata records semantic policy `v2` and in-process execution.

## XLM-R checkpoint placement

The newly trained checkpoint was moved to `models/reader/xlm-roberta-large-viquad`, which is the existing `ReaderManager` target for reader name `xlmr`. It contains `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and `training_args.bin`. A training-export compatibility field (`extra_special_tokens` encoded as a list) was removed from tokenizer metadata; offline `AutoConfig` and `AutoTokenizer` loading then succeeded as `XLMRobertaForQuestionAnswering` and `XLMRobertaTokenizerFast` with vocabulary size 250,002. Safetensors header validation found 391 tensors, including `qa_outputs.weight` and `qa_outputs.bias`.

The semantic holdout remained on the frozen PhoBERT checkpoint as required; installing XLM-R did not silently change benchmark model, BM25, top_k, or global reader threshold.

## File inventory

- Backend/config: `backend/config.py`, `backend/viqa_api.py`, `config/qa_pipeline.json`, `config/semantic_policy.json`.
- Semantic core: `reader/semantic_policy.py`, `reader/candidate_validation.py`, `reader/relation_validator.py`, `reader/rejection_reasons.py`, `reader/subject_consistency.py`, `reader/answer_completeness.py`, `reader/span_boundaries.py`, `reader/candidates.py`.
- Parsing/fallback/shared utilities: `reader/question_semantics.py`, `reader/fallback_extractor.py`, `reader/fallback_policy.py`, `reader/cause_relations.py`, `reader/lexical.py`.
- Evaluation: `build_semantic_holdout.py`, `evaluate_semantic_holdout.py`, `evaluate_semantic_rules.py`, `data/evaluation/semantic_holdout_v1.jsonl`, its lock metadata, and reports/caches under `results/`.
- Tests: `tests/test_semantic_architecture.py`, `tests/test_semantic_holdout.py`.
- Model artifact: `models/reader/xlm-roberta-large-viquad/tokenizer_config.json` plus the moved trained checkpoint files.

## Verification

- `python -m pytest -q`: 119 passed.
- Architecture/holdout focused tests: 11 passed.
- `git diff --check`: no whitespace error; only Windows CRLF conversion warnings.
- Production branch search: no candidate relation/method equality branch; remaining method equality branches choose retrievers only.
- Proper-name and fixture-import searches: no matches in production/config.
- Final holdout: 260 rows, 260 unique IDs, ablation ready, all candidate gate traces present.
