# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/train_sample_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s3_top4_hard_lr/train_sample/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s3_top4_hard_lr/train_sample/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 1,000 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,000 / 1,000 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 88.600% (886 / 1,000) |
| Relaxed Accuracy | 88.600% (886 / 1,000) |
| Target Kana-CER | 6.745% |
| Target Kana-CER@1 | 6.512% |
| Relaxed Target Kana-CER | 6.745% |
| Relaxed Target Kana-CER@1 | 6.512% |
| Sentence Kana-CER | 1.623% |
| Sentence Kana-CER@1 | 1.623% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 441 | 86.621% | 86.621% | 8.455% | 8.153% | 8.455% | 8.153% |
| Kun’yomi | 409 | 92.910% | 92.910% | 4.014% | 4.014% | 4.014% | 4.014% |
| Jōyō appendix readings | 150 | 82.667% | 82.667% | 9.167% | 8.500% | 9.167% | 8.500% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 1,000 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 114 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 267 | Rows with nonzero Sentence Kana-CER |
