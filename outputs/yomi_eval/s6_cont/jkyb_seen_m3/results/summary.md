# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/jkyb_seen_m3_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s6_cont/jkyb_seen_m3/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s6_cont/jkyb_seen_m3/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 864 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 864 / 864 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 89.815% (776 / 864) |
| Relaxed Accuracy | 89.931% (777 / 864) |
| Target Kana-CER | 8.341% |
| Target Kana-CER@1 | 7.415% |
| Relaxed Target Kana-CER | 8.225% |
| Relaxed Target Kana-CER@1 | 7.299% |
| Sentence Kana-CER | 1.381% |
| Sentence Kana-CER@1 | 1.381% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 488 | 89.344% | 89.344% | 9.529% | 8.402% | 9.529% | 8.402% |
| Kun’yomi | 246 | 92.276% | 92.276% | 5.366% | 5.366% | 5.366% | 5.366% |
| Jōyō appendix readings | 130 | 86.923% | 87.692% | 9.513% | 7.590% | 8.744% | 6.821% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 864 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 88 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 169 | Rows with nonzero Sentence Kana-CER |
