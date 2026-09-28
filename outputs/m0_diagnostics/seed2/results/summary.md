# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/subset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/seed2/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/seed2/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/seed2/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 1,764 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,764 / 1,764 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 58.277% (1,028 / 1,764) |
| Relaxed Accuracy | 59.070% (1,042 / 1,764) |
| Target Kana-CER | 39.764% |
| Target Kana-CER@1 | 33.868% |
| Relaxed Target Kana-CER | 38.918% |
| Relaxed Target Kana-CER@1 | 33.117% |
| Sentence Kana-CER | 4.269% |
| Sentence Kana-CER@1 | 4.269% |
| Text CER | 6.645% |
| Text CER@1 | 6.642% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 991 | 53.380% | 53.683% | 44.299% | 38.009% | 44.046% | 37.756% |
| Kun’yomi | 683 | 67.643% | 68.668% | 32.391% | 27.438% | 31.220% | 26.413% |
| Jōyō appendix readings | 90 | 41.111% | 45.556% | 45.778% | 37.074% | 40.870% | 32.907% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 1 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 1,764 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 736 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 826 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 912 | Rows with nonzero Text CER |
