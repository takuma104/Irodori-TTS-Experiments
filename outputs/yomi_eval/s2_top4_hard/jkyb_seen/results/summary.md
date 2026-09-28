# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/jkyb_seen_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s2_top4_hard/jkyb_seen/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s2_top4_hard/jkyb_seen/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 316 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 316 / 316 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 84.494% (267 / 316) |
| Relaxed Accuracy | 84.810% (268 / 316) |
| Target Kana-CER | 13.370% |
| Target Kana-CER@1 | 11.472% |
| Relaxed Target Kana-CER | 13.054% |
| Relaxed Target Kana-CER@1 | 11.155% |
| Sentence Kana-CER | 2.046% |
| Sentence Kana-CER@1 | 2.046% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 163 | 82.209% | 82.209% | 16.462% | 14.928% | 16.462% | 14.928% |
| Kun’yomi | 33 | 100.000% | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| Jōyō appendix readings | 120 | 83.333% | 84.167% | 12.847% | 9.931% | 12.014% | 9.097% |

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
| [Target review](details/target-review.jsonl) | 49 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 72 | Rows with nonzero Sentence Kana-CER |
