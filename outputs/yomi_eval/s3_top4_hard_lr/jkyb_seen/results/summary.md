# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/jkyb_seen_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s3_top4_hard_lr/jkyb_seen/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s3_top4_hard_lr/jkyb_seen/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 316 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 316 / 316 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 91.456% (289 / 316) |
| Relaxed Accuracy | 91.772% (290 / 316) |
| Target Kana-CER | 5.960% |
| Target Kana-CER@1 | 5.485% |
| Relaxed Target Kana-CER | 5.643% |
| Relaxed Target Kana-CER@1 | 5.169% |
| Sentence Kana-CER | 1.238% |
| Sentence Kana-CER@1 | 1.238% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 163 | 91.411% | 91.411% | 6.339% | 5.726% | 6.339% | 5.726% |
| Kun’yomi | 33 | 96.970% | 96.970% | 3.030% | 3.030% | 3.030% | 3.030% |
| Jōyō appendix readings | 120 | 90.000% | 90.833% | 6.250% | 5.833% | 5.417% | 5.000% |

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
| [Target review](details/target-review.jsonl) | 27 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 60 | Rows with nonzero Sentence Kana-CER |
