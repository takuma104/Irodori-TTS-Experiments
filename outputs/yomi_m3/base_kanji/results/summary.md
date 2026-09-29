# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_m3/base_kanji/dataset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_m3/base_kanji/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_m3/base_kanji/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 78,595 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 78,595 / 78,595 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 68.920% (54,168 / 78,595) |
| Relaxed Accuracy | 68.920% (54,168 / 78,595) |
| Target Kana-CER | 27.926% |
| Target Kana-CER@1 | 23.641% |
| Relaxed Target Kana-CER | 27.926% |
| Relaxed Target Kana-CER@1 | 23.641% |
| Sentence Kana-CER | 3.731% |
| Sentence Kana-CER@1 | 3.731% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 42,443 | 66.477% | 66.477% | 30.212% | 25.514% | 30.212% | 25.514% |
| Kun’yomi | 30,581 | 73.209% | 73.209% | 23.426% | 20.104% | 23.426% | 20.104% |
| Jōyō appendix readings | 5,571 | 63.992% | 63.992% | 35.215% | 28.787% | 35.215% | 28.787% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 2 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 78,595 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 24,427 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 35,947 | Rows with nonzero Sentence Kana-CER |
