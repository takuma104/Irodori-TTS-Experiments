# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_pilot/base_kanji/dataset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_pilot/base_kanji/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_pilot/base_kanji/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 6,972 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 6,972 / 6,972 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 70.496% (4,915 / 6,972) |
| Relaxed Accuracy | 70.496% (4,915 / 6,972) |
| Target Kana-CER | 25.320% |
| Target Kana-CER@1 | 21.674% |
| Relaxed Target Kana-CER | 25.320% |
| Relaxed Target Kana-CER@1 | 21.674% |
| Sentence Kana-CER | 3.538% |
| Sentence Kana-CER@1 | 3.538% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 3,644 | 68.825% | 68.825% | 25.507% | 22.109% | 25.507% | 22.109% |
| Kun’yomi | 2,745 | 77.632% | 77.632% | 19.481% | 16.788% | 19.481% | 16.788% |
| Jōyō appendix readings | 583 | 47.341% | 47.341% | 51.644% | 41.961% | 51.644% | 41.961% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 6,972 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 2,057 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 2,963 | Rows with nonzero Sentence Kana-CER |
