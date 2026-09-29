# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s7_cont/jkyb/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s7_cont/jkyb/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s7_cont/jkyb/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 13,536 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 13,536 / 13,536 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 94.644% (12,811 / 13,536) |
| Relaxed Accuracy | 94.733% (12,823 / 13,536) |
| Target Kana-CER | 4.715% |
| Target Kana-CER@1 | 4.118% |
| Relaxed Target Kana-CER | 4.618% |
| Relaxed Target Kana-CER@1 | 4.033% |
| Sentence Kana-CER | 0.845% |
| Sentence Kana-CER@1 | 0.845% |
| Text CER | 3.998% |
| Text CER@1 | 3.996% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 7,056 | 94.033% | 94.062% | 5.378% | 4.712% | 5.357% | 4.691% |
| Kun’yomi | 6,108 | 95.825% | 95.940% | 3.670% | 3.203% | 3.539% | 3.089% |
| Jōyō appendix readings | 372 | 86.828% | 87.634% | 9.310% | 7.876% | 8.324% | 7.070% |

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
| [Target review](details/target-review.jsonl) | 725 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,629 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 4,582 | Rows with nonzero Text CER |
