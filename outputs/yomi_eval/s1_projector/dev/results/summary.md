# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/dev_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s1_projector/dev/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s1_projector/dev/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 898 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 898 / 898 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 79.733% (716 / 898) |
| Relaxed Accuracy | 79.733% (716 / 898) |
| Target Kana-CER | 15.657% |
| Target Kana-CER@1 | 14.761% |
| Relaxed Target Kana-CER | 15.657% |
| Relaxed Target Kana-CER@1 | 14.761% |
| Sentence Kana-CER | 2.720% |
| Sentence Kana-CER@1 | 2.720% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 535 | 79.065% | 79.065% | 16.841% | 15.720% | 16.841% | 15.720% |
| Kun’yomi | 319 | 85.580% | 85.580% | 9.963% | 9.650% | 9.963% | 9.650% |
| Jōyō appendix readings | 44 | 45.455% | 45.455% | 42.538% | 40.152% | 42.538% | 40.152% |

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
| [Target review](details/target-review.jsonl) | 182 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 308 | Rows with nonzero Sentence Kana-CER |
