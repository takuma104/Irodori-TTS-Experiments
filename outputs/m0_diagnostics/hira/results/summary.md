# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/subset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/hira/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/hira/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/m0_diagnostics/hira/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 1,764 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,764 / 1,764 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 89.456% (1,578 / 1,764) |
| Relaxed Accuracy | 89.456% (1,578 / 1,764) |
| Target Kana-CER | 8.007% |
| Target Kana-CER@1 | 7.440% |
| Relaxed Target Kana-CER | 8.007% |
| Relaxed Target Kana-CER@1 | 7.440% |
| Sentence Kana-CER | 1.204% |
| Sentence Kana-CER@1 | 1.204% |
| Text CER | 7.282% |
| Text CER@1 | 7.271% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 991 | 88.597% | 88.597% | 9.284% | 8.274% | 9.284% | 8.274% |
| Kun’yomi | 683 | 90.483% | 90.483% | 6.686% | 6.686% | 6.686% | 6.686% |
| Jōyō appendix readings | 90 | 91.111% | 91.111% | 3.981% | 3.981% | 3.981% | 3.981% |

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
| [Target review](details/target-review.jsonl) | 186 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 347 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 989 | Rows with nonzero Text CER |
