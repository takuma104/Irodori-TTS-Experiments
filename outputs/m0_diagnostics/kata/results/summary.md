# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/subset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/kata/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/kata/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/kata/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 1,764 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,764 / 1,764 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 90.079% (1,589 / 1,764) |
| Relaxed Accuracy | 90.136% (1,590 / 1,764) |
| Target Kana-CER | 7.615% |
| Target Kana-CER@1 | 6.822% |
| Relaxed Target Kana-CER | 7.559% |
| Relaxed Target Kana-CER@1 | 6.765% |
| Sentence Kana-CER | 1.154% |
| Sentence Kana-CER@1 | 1.154% |
| Text CER | 7.178% |
| Text CER@1 | 7.167% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 991 | 88.799% | 88.799% | 9.452% | 8.039% | 9.452% | 8.039% |
| Kun’yomi | 683 | 92.240% | 92.387% | 5.234% | 5.234% | 5.088% | 5.088% |
| Jōyō appendix readings | 90 | 87.778% | 87.778% | 5.463% | 5.463% | 5.463% | 5.463% |

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
| [Target review](details/target-review.jsonl) | 175 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 341 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 977 | Rows with nonzero Text CER |
