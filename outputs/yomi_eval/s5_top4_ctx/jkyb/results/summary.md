# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s5_top4_ctx/jkyb/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s5_top4_ctx/jkyb/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s5_top4_ctx/jkyb/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 13,536 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 13,536 / 13,536 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 94.297% (12,764 / 13,536) |
| Relaxed Accuracy | 94.385% (12,776 / 13,536) |
| Target Kana-CER | 5.173% |
| Target Kana-CER@1 | 4.477% |
| Relaxed Target Kana-CER | 5.076% |
| Relaxed Target Kana-CER@1 | 4.392% |
| Sentence Kana-CER | 0.873% |
| Sentence Kana-CER@1 | 0.873% |
| Text CER | 3.994% |
| Text CER@1 | 3.991% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 7,056 | 93.339% | 93.367% | 6.113% | 5.300% | 6.092% | 5.279% |
| Kun’yomi | 6,108 | 95.760% | 95.874% | 3.855% | 3.334% | 3.724% | 3.219% |
| Jōyō appendix readings | 372 | 88.441% | 89.247% | 8.983% | 7.639% | 7.997% | 6.832% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 13,536 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 772 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,627 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 4,558 | Rows with nonzero Text CER |
