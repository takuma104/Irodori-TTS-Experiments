# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/jkyb_seen_m3_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s5_top4_ctx/jkyb_seen_m3/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s5_top4_ctx/jkyb_seen_m3/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 864 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 864 / 864 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 87.963% (760 / 864) |
| Relaxed Accuracy | 88.079% (761 / 864) |
| Target Kana-CER | 10.525% |
| Target Kana-CER@1 | 9.367% |
| Relaxed Target Kana-CER | 10.409% |
| Relaxed Target Kana-CER@1 | 9.252% |
| Sentence Kana-CER | 1.651% |
| Sentence Kana-CER@1 | 1.651% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 488 | 85.861% | 85.861% | 12.534% | 11.202% | 12.534% | 11.202% |
| Kun’yomi | 246 | 91.870% | 91.870% | 6.748% | 6.341% | 6.748% | 6.341% |
| Jōyō appendix readings | 130 | 88.462% | 89.231% | 10.128% | 8.205% | 9.359% | 7.436% |

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
| [Target review](details/target-review.jsonl) | 104 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 180 | Rows with nonzero Sentence Kana-CER |
