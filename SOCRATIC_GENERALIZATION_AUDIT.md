# Socratic Follow-up Generalization Audit

Date: 2026-08-21  
Scope: Socratic discovery, generation, validation, ranking, diagnostics, and frontend development diagnostics.  
Main QA constraint: `/api/ask`, BM25, Reader inference, main semantic validation, answer refinement, and main reranking were not changed.

## Executive result

The pre-task runtime was not entity-hard-coded, and it already discovered evidence before rendering a template. Its generalization bottleneck was narrower: several typed opportunity detectors still depended on enumerated action predicates, person-oriented coreference, a role-title whitelist, and a height-only attribute rule. Biography-style passages therefore exposed many recognized opportunities while unseen predicates and non-biography passages frequently fell through to zero or one generic candidate.

The refactor preserves evidence-first generation and makes the typed discovery layer structural:

```text
Current QA semantics
  -> selected + retrieved corpus passages
  -> subject/coreference binding
  -> structural semantic fact discovery
  -> KnowledgeOpportunity
  -> grounded question rendering
  -> relation/evidence/subject/topic/answerability/novelty gates
  -> diversity-aware ranking
  -> top 1-3 follow-ups
```

No production rule contains an evaluation entity, question, passage ID, expected answer, or import from `tests/data`.

### Post-audit product-policy override: always-on tutoring

After the locked evaluation above was completed, the product requirement changed from “an empty result is allowed when no novel fact survives” to “every QA turn must receive a follow-up suggestion.” The runtime now applies this contract whenever `/api/ask` returns at least one selected or retrieved corpus passage:

- verified, novel, typed knowledge questions remain the first choice;
- a missing main answer no longer prevents Socratic generation;
- if all typed candidates are exhausted or rejected, the service returns one `EVIDENCE_DETAIL` review question bound to a real passage, `source_passage_id`, and exact `evidence_sentence`;
- if retrieval misses the question subject, the fallback pivots to the retrieved document's real title instead of claiming unsupported evidence about the requested subject.

The locked before/after artifacts and metrics below remain the historical evaluation of the earlier sparse-output policy. They were not overwritten or tuned after inspection. Under the current product policy, a zero-result outcome is retained only when no corpus passage exists at all.

## Locked generalization holdout

- Dataset: `tests/data/socratic_generalization_holdout_v1.json`
- Lock: `tests/data/socratic_generalization_holdout_v1.lock.json`
- Status: `LOCKED_DO_NOT_TUNE`
- Cases: 80 real validation-corpus cases
- Checksum: `a72e450b2ce1f47c4b0c2508c738fbfa7945d2e89afb35cdd7cb7de711a99ea7`
- Unique article titles: 15
- Strata: 10 cases each for `IDENTITY_DEFINITION`, `TIME`, `LOCATION`, `CAUSE`, `PURPOSE`, `ENTITY`, `ATTRIBUTE_GENERAL`, and `NO_ANSWER_SPARSE`
- Exclusions: all named benchmark entities in the task were excluded before selection.
- Tuning policy: no individual holdout failure was inspected or used to patch production. Only aggregate final evaluation is reported. The weak opportunity labels are diagnostic proxies, not human gold.

The answer sentence is the selected passage; other sentences from the same real validation context are separate retrieved passages. This makes selected-only versus selected+retrieved behavior measurable without fabricating evidence.

## Before/after metrics

Primary comparison uses selected plus retrieved passages.

| Metric | Before | After |
|---|---:|---:|
| Cases with follow-up | 45/80 | 53/80 |
| Raw coverage | 56.25% | 66.25% |
| Opportunity-aware coverage | 64.29% | 75.71% |
| Weak opportunity recall | 10.33% | 20.33% |
| Average follow-ups/case | 0.700 | 1.225 |
| Empty rate | 43.75% | 33.75% |
| Grounding proxy | 100% | 100% |
| Answerability proxy | 100% | 100% |
| Duplicate rate | 0% | 0% |
| Off-topic rate | 0% | 0% |
| Mean latency | 4.191 ms | 5.366 ms |
| P50 latency | 4.072 ms | 4.922 ms |
| P95 latency | 8.298 ms | 11.288 ms |

Coverage by current-question stratum:

| Stratum | Before | After |
|---|---:|---:|
| IDENTITY / DEFINITION | 60% | 60% |
| TIME | 70% | 90% |
| LOCATION | 70% | 90% |
| CAUSE | 70% | 80% |
| PURPOSE | 60% | 70% |
| ENTITY | 50% | 60% |
| ATTRIBUTE / GENERAL | 70% | 80% |
| NO-ANSWER / SPARSE | 0% | 0% (historical pre-override result) |

Post-refactor weak opportunity recall by available relation:

| Relation family | Known available cases | Discovered cases | Final accepted cases | Recall |
|---|---:|---:|---:|---:|
| ATTRIBUTE | 25 | 1 | 1 | 4.00% |
| CAUSE / CONSEQUENCE | 49 | 5 | 4 | 10.20% |
| EVENT | 26 | 8 | 8 | 30.77% |
| IDENTITY / DEFINITION | 44 | 2 | 2 | 4.55% |
| LOCATION | 51 | 11 | 11 | 21.57% |
| PURPOSE | 34 | 9 | 9 | 26.47% |
| ROLE | 19 | 2 | 2 | 10.53% |
| TIME | 52 | 23 | 23 | 44.23% |

These relation annotations are intentionally broad weak labels. Low recall is reported rather than hidden; it is not safe to convert individual v1 misses into new production patterns while retaining the holdout label.

## Runtime hard-code classification

### A. Answer hard-code

None found. There is no `question/entity -> fixed answer`, no expected answer in templates, and no production import of regression or holdout fixtures.

### B. Case hard-code

None found. Search across the Socratic runtime found:

- production entity literals from the prohibited benchmark list: 0
- production question literals from benchmark cases: 0
- production passage IDs: 0
- entity- or passage-specific branches: 0

### C. Generic linguistic rules

Present and permitted. They describe relation evidence such as causal connectors, purpose connectors, explicit time values, locative connectors, appointment grammar, activity grammar, and local coreference. All matching is token-boundary/context-bound; no blind substring rule is used for `sinh`, `để`, or ambiguous accent-fold collisions.

The following narrow lists were removed/refactored:

- `_SPATIAL_PROCESS` exact predicate list: removed; the predicate is now extracted from `subject + predicate + tại/ở + location` evidence.
- `_spatial_process_predicate` display whitelist: removed.
- `_TEMPORAL_ACTION` exact action list: removed; event time uses a subject-bound clause plus explicit temporal evidence.
- `role_noun` title whitelist: removed; role detection uses appointment/office grammar and accepts unseen role titles.

Generic relation cue sets such as causal/purpose markers remain linguistic rules, not benchmark predicates.

## KnowledgeOpportunity

`KnowledgeOpportunity` is created only after a sentence is bound to the current subject (direct mention, local coreference, or title-anchored coreference) and contains typed evidence. It records:

- subject, relation, object/target, predicate
- evidence sentence and source passage ID
- subject, relation, evidence, topic, and retrieval relevance scores
- selected/retrieved origin and discovery rule
- question type and provenance

`FollowUpOpportunity` remains only as a compatibility alias; runtime instances are `KnowledgeOpportunity` objects. Templates run after opportunity discovery and never search for evidence to justify a preselected question.

## Passage scanning and coreference

- The selected passage is scanned first, followed by top retrieved passages.
- The limit is `max_passages_for_followup_discovery` in `config/socratic.json` (currently 12).
- Ranking does not grant selected evidence an unconditional win; retrieved opportunities can outrank it on evidence, topic, novelty, and diversity.
- Subject consistency reuses `reader.subject_consistency.score_subject_consistency`.
- Socratic-local title anchoring extends coreference to people, organizations, countries, events, places, areas, objects, and constructions without changing main QA validation.

## Gates, diagnostics, and empty outcomes

All Socratic thresholds and limits are in `config/socratic.json`: subject, relation, evidence, topic, answerability, novelty, ranking, duplicate similarity, maximum internal candidates, maximum passages, probes, and final count.

Debug output now includes inspected passage IDs, opportunity counts by relation, generated candidates, stage counts, rejection distribution, evidence, origin, final accepted count, and latency. Empty results distinguish:

- `NO_OTHER_KNOWLEDGE_IN_CONTEXT`
- `OPPORTUNITIES_FOUND_BUT_GENERATOR_FAILED`
- `CANDIDATES_FOUND_BUT_GATES_TOO_STRICT`
- `ONLY_CURRENT_RELATION_AVAILABLE`

The development UI requests and renders these diagnostics. Production builds do not request/show the panel.

## Rule coverage and overfit review

The full report is `results/socratic_rule_coverage.json`. One rule is flagged by the required mechanical criterion:

- `structural_numeric_attribute`: `POSSIBLE_CASE_OVERFIT` because it matched one holdout case with one subject.

It was not removed: the rule is a generic measure-predicate plus numeric-unit grammar and has dedicated contrast tests. Low empirical coverage is a review flag, not proof of case-specific logic. `structural_process_location` matched 11 cases, 7 subjects, and 10 distinct predicates, demonstrating that it is no longer tied to a known predicate list.

## Rejection analysis

For selected plus retrieved passages after the refactor:

- `FOLLOWUPS_GENERATED`: 53 cases
- `NO_OTHER_KNOWLEDGE_IN_CONTEXT`: 16 cases
- `ONLY_CURRENT_RELATION_AVAILABLE`: 1 case
- `INPUT_NOT_ELIGIBLE`: 10 no-answer cases
- candidate rejections: `LOW_RANKING_SCORE` 11, `SAME_RELATION` 10, `DUPLICATE_QUESTION` 4

Full per-case stage data is in `results/socratic_rejection_analysis.json`.

## Human review

The after artifact contains a deterministic queue of 50 grounded follow-ups with the original question, evidence sentence, source passage, relation, and blank review fields for `GROUNDED`, `RELEVANT`, `NOVEL`, `ANSWERABLE`, and `NATURAL`.

Status: `NOT_PERFORMED_REQUIRES_INDEPENDENT_REVIEWER`.

No automated proxy is represented as a completed human review or as 100% human quality. If a reviewer uses individual v1 failures for development, v1 must be marked `CONSUMED_FOR_DEVELOPMENT` and a new locked v2 must be sampled.

## Required question-by-question summary

1. Runtime question/entity hard-code: **No**.
2. Benchmark-specific predicates: **No** after refactor.
3. Overfit flags: `structural_numeric_attribute` is mechanically flagged for review; no case literal was found.
4. Root cause of known biography passing while other questions failed: narrow typed predicate/role/time detectors and person-centric coreference, not an entity answer map.
5. Old baseline generator start: evidence-first skeleton, but with narrow enumerated typed detectors; templates rendered afterward.
6. New generator start: selected/retrieved evidence sentences and subject/coreference binding.
7. Opportunity construction: a structurally detected, subject-bound, scored fact becomes `KnowledgeOpportunity`.
8. Retrieved passages scanned: **Yes**, config-limited.
9. Coreference module: `reader.subject_consistency.score_subject_consistency`, plus local title-anchored categories.
10. Removed/refactored lists: spatial-process predicates, temporal-action predicates, and role nouns.
11. Holdout size: **80**.
12. Holdout strata: eight groups listed above.
13. Raw coverage before: **56.25%**.
14. Raw coverage after: **66.25%**.
15. Opportunity-aware coverage after: **75.71%**.
16. Coverage by relation: see table above.
17. Average follow-ups: **0.700 -> 1.225**.
18. Empty rate: **43.75% -> 33.75%**.
19. Duplicate rate: **0% -> 0%**.
20. Off-topic rate: **0% -> 0%**.
21. Answerability proxy: **100% -> 100%**.
22. Latency after: **P50 4.922 ms, P95 11.288 ms** for selected+retrieved.
23. Human review: 50-item queue created; independent human review not yet performed.
24. Known regression: all Socratic regression cases pass; the broader frozen main-QA suite has seven out-of-scope failures recorded below.
25. Full Socratic test count after the always-on override: **55 passed**.
26. Production files changed: `backend/socratic.py`, `config/socratic.json`, and development-only frontend diagnostics/types/hooks/components.
27. Holdout used for tuning: **No**; no individual v1 failure was inspected or patched.

## Artifacts

- `results/socratic_generalization_before.json`
- `results/socratic_generalization_after.json`
- `results/socratic_rule_coverage.json`
- `results/socratic_rejection_analysis.json`
- `SOCRATIC_GENERALIZATION_AUDIT.md`

## Verification

- `python -m pytest tests/test_socratic.py tests/test_socratic_generalization.py tests/test_socratic_ui.py -q`: **55 passed**.
- Socratic known regressions, new contrast/generalization tests, locked-holdout contract, endpoint contract, always-on fallback, and frontend contract are included in those 55 tests.
- `npm run build`: **passed** (the existing Vite large-chunk warning remains).
- `python -m py_compile backend/socratic.py build_socratic_generalization_holdout.py evaluate_socratic_generalization.py`: **passed**.
- `git diff --check`: **passed**.
- Live HTTP smoke test on a corpus question: main answer returned first; Socratic then inspected 10 passages, found 2 opportunities, accepted 2 follow-ups, and every returned item contained `source_passage_id`, `evidence_sentence`, and `relation_evidence=true`.
- Broader semantic/main-QA command: **123 passed, 7 failed**. The failures are in frozen `/api/ask` behavior (`test_live_semantic_regressions.py` and `test_pipeline.py`) covering cause rejection, question typing, fallback rejection details, reranking weights, and fallback answer extraction. None executes Socratic generation, and no implicated main-QA production file was changed in this task. They are reported rather than silently patched because the task explicitly forbids modifying main QA.

Automated quality metrics remain proxies and should be paired with the queued independent review before making a human-quality claim.
