# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/jkyb_seen_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s4_top4_ctx/jkyb_seen/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s4_top4_ctx/jkyb_seen/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 316 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 316 / 316 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 89.873% (284 / 316) |
| Relaxed Accuracy | 90.190% (285 / 316) |
| Target Kana-CER | 9.652% |
| Target Kana-CER@1 | 7.911% |
| Relaxed Target Kana-CER | 9.335% |
| Relaxed Target Kana-CER@1 | 7.595% |
| Sentence Kana-CER | 1.489% |
| Sentence Kana-CER@1 | 1.489% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 163 | 88.344% | 88.344% | 11.861% | 9.714% | 11.861% | 9.714% |
| Kun’yomi | 33 | 96.970% | 96.970% | 1.515% | 1.515% | 1.515% | 1.515% |
| Jōyō appendix readings | 120 | 90.000% | 90.833% | 8.889% | 7.222% | 8.056% | 6.389% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 316 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 32 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 57 | Rows with nonzero Sentence Kana-CER |
