# New Reader audit and calibration report

## Decision

- Checkpoint is retained at `models/reader/heidiie_phobert_finetuned_viquad`.
- Status: **experimental**, not production-ready.
- Reader threshold: `-3.5`, selected on all 3,814 validation examples with
  `0.7 × Answerable F1 + 0.3 × Unanswerable accuracy`.
- Active experimental reranking: Retriever `0.40`, Reader `0.30`, Answer Type
  `0.20`, Relation `0.10`; final gate `0.625` from a stratified 100-example
  candidate-pipeline benchmark.
- Phrase fallback penalty is `1.0`; whole-sentence fallback remains penalized at `0.6`.
- Reader inference preserves the top 5 valid spans per passage. Boundary-aware
  reranking penalizes shifted/incomplete spans without blindly extending them.

## Checkpoint audit

The checkpoint loads successfully through `AutoModelForQuestionAnswering` as
`RobertaForQuestionAnswering`. Its `PhobertTokenizer` and model both have a
64,001-token vocabulary. QA-head weights and bias are present with shapes
`[2, 768]` and `[2]`; loading reported no missing or unexpected keys.

Verified training arguments from `training_args.bin`: 3 epochs, learning rate
`2e-5`, train/eval batch size 16, gradient accumulation 1, weight decay `0.01`,
linear scheduler, no warmup, FP16, seed 42. `load_best_model_at_end=false` and
`metric_for_best_model=null`.

The artifact does not encode the original training max length, stride, custom
versus default loss, train/validation sample counts, or a post-processed
checkpoint selection metric. These values remain unknown rather than inferred.

## Full Reader validation comparison

| Metric | Old | New | New - old |
|---|---:|---:|---:|
| Overall EM | 27.87 | 13.87 | -14.00 |
| Overall F1 | 38.03 | 36.25 | -1.78 |
| Answerable EM | 14.47 | 0.79 | -13.68 |
| Answerable F1 | 29.08 | 32.96 | +3.88 |
| Unanswerable accuracy | 58.48 | 42.38 | -16.11 |
| Answerable predicted-empty rate | 43.61 | 34.83 | -8.78 |
| Selected Reader threshold | -1.8 | -3.5 | — |

The new Reader lowers false negatives but has substantial span-boundary and
false-positive regressions. It is therefore retained for experimentation but
is not promoted as better than the old checkpoint.

## New checkpoint score-margin distribution

| Class | Mean | Median | Std | P10 | P25 | P50 | P75 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Answerable | -2.376 | -1.902 | 3.115 | -6.836 | -4.614 | -1.902 | 0.133 | 1.381 |
| Unanswerable | -2.973 | -2.546 | 3.157 | -7.435 | -5.326 | -2.546 | -0.355 | 0.889 |

The overlap explains why threshold `0.0` rejected many useful spans and why a
single margin cannot be treated as a correctness probability.

## Threshold objectives for the new Reader

| Objective | Threshold | Overall F1 | Answerable F1 | Unanswerable accuracy | Answerable empty rate |
|---|---:|---:|---:|---:|---:|
| Max overall F1 | -3.801599 | 36.25 | 34.32 | 39.10 | 31.85 |
| Max answerable F1 | -10.0 | 33.20 | 45.95 | 1.64 | 0.98 |
| 0.7 answerable F1 + 0.3 unanswerable accuracy | -3.5 | 36.25 | 32.96 | 42.38 | 34.83 |

At `-1.8`, the new Reader reaches Answerable F1 `25.82`, unanswerable accuracy
`58.40`, and answerable empty rate `51.11`; it is not the selected threshold.

## Reproduced parent/software case

The real BM25 pipeline retrieves `doc_00031_P0001`. The neural candidate is
retained with margin `-1.276181` and ranking score `0.784507`. A complete
contrast-aware phrase fallback scores `0.81` and wins final ranking:

> sự khác biệt giữa các chức năng học thực sự (như ở đây) và phần mềm thiên về
> mặt giáo dục (được trình bay ở sau)

The neural candidate is not deleted before fallback. Both proposals, their
gates, margins, evidence, relation scores, and rejection reasons are exposed in
the development response.

## Verification

- 66 unit/regression tests pass, including Paris TIME, Saint-Pierre ENTITY,
  French Revolution LOCATION, false premise, below-threshold candidate
  preservation, and fallback override behavior.
- TypeScript/Vite production build succeeds.
- Backend `/health` returns `ok`; frontend returns HTTP 200.
- Real `/api/ask` reproduces the corrected answer above.

## Remaining promotion gate

Full 3,814-example **Reader** evaluation is complete for both checkpoints. A
full 3,814-example **end-to-end Top-10/20-candidate** run is not complete on this
CPU-only host; the measured 100-example stratified run takes about 6.9 minutes,
which projects to roughly 4.4 hours. Its candidate cache and evaluator are
versioned so the full run can be executed on Kaggle GPU before promotion.
