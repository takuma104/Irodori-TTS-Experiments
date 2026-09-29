# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/jkyb_seen_m3_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/jkyb_seen_m3/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/jkyb_seen_m3/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 864 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 864 / 864 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 86.574% (748 / 864) |
| Relaxed Accuracy | 86.690% (749 / 864) |
| Target Kana-CER | 11.516% |
| Target Kana-CER@1 | 10.127% |
| Relaxed Target Kana-CER | 11.400% |
| Relaxed Target Kana-CER@1 | 10.012% |
| Sentence Kana-CER | 1.692% |
| Sentence Kana-CER@1 | 1.692% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 488 | 84.221% | 84.221% | 13.695% | 12.466% | 13.695% | 12.466% |
| Kun’yomi | 246 | 91.463% | 91.463% | 6.626% | 6.220% | 6.626% | 6.220% |
| Jōyō appendix readings | 130 | 86.154% | 86.923% | 12.590% | 8.744% | 11.821% | 7.974% |

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
| [Target review](details/target-review.jsonl) | 116 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 184 | Rows with nonzero Sentence Kana-CER |
