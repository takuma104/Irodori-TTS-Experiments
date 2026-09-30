# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/dev_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/dev/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/dev/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 898 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 898 / 898 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 82.517% (741 / 898) |
| Relaxed Accuracy | 82.517% (741 / 898) |
| Target Kana-CER | 13.469% |
| Target Kana-CER@1 | 12.767% |
| Relaxed Target Kana-CER | 13.469% |
| Relaxed Target Kana-CER@1 | 12.767% |
| Sentence Kana-CER | 2.546% |
| Sentence Kana-CER@1 | 2.546% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 535 | 82.430% | 82.430% | 12.925% | 12.364% | 12.925% | 12.364% |
| Kun’yomi | 319 | 86.207% | 86.207% | 11.076% | 10.293% | 11.076% | 10.293% |
| Jōyō appendix readings | 44 | 56.818% | 56.818% | 37.424% | 35.606% | 37.424% | 35.606% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 898 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 157 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 290 | Rows with nonzero Sentence Kana-CER |
