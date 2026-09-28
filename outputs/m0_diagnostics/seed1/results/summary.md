# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/subset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/seed1/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/seed1/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/seed1/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 1,764 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,764 / 1,764 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 58.560% (1,033 / 1,764) |
| Relaxed Accuracy | 59.240% (1,045 / 1,764) |
| Target Kana-CER | 39.563% |
| Target Kana-CER@1 | 33.488% |
| Relaxed Target Kana-CER | 38.817% |
| Relaxed Target Kana-CER@1 | 32.836% |
| Sentence Kana-CER | 4.184% |
| Sentence Kana-CER@1 | 4.184% |
| Text CER | 6.738% |
| Text CER@1 | 6.738% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 991 | 53.683% | 53.986% | 44.383% | 37.740% | 44.131% | 37.487% |
| Kun’yomi | 683 | 68.228% | 69.107% | 32.094% | 26.920% | 31.069% | 26.042% |
| Jōyō appendix readings | 90 | 38.889% | 42.222% | 43.185% | 36.519% | 39.111% | 33.185% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 1,764 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 731 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 805 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 895 | Rows with nonzero Text CER |
