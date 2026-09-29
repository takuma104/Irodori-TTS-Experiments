# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/train_hard_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s7_cont/train_hard/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s7_cont/train_hard/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 1,200 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,200 / 1,200 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 58.333% (700 / 1,200) |
| Relaxed Accuracy | 58.333% (700 / 1,200) |
| Target Kana-CER | 30.760% |
| Target Kana-CER@1 | 27.510% |
| Relaxed Target Kana-CER | 30.760% |
| Relaxed Target Kana-CER@1 | 27.510% |
| Sentence Kana-CER | 4.079% |
| Sentence Kana-CER@1 | 4.079% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 510 | 60.588% | 60.588% | 28.518% | 25.380% | 28.518% | 25.380% |
| Kun’yomi | 490 | 63.061% | 63.061% | 26.791% | 24.444% | 26.791% | 24.444% |
| Jōyō appendix readings | 200 | 41.000% | 41.000% | 46.202% | 40.452% | 46.202% | 40.452% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 1,200 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 500 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 682 | Rows with nonzero Sentence Kana-CER |
