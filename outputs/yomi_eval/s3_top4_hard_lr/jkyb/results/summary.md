# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s3_top4_hard_lr/jkyb/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s3_top4_hard_lr/jkyb/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s3_top4_hard_lr/jkyb/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 13,536 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 13,536 / 13,536 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 93.477% (12,653 / 13,536) |
| Relaxed Accuracy | 93.595% (12,669 / 13,536) |
| Target Kana-CER | 5.968% |
| Target Kana-CER@1 | 5.112% |
| Relaxed Target Kana-CER | 5.842% |
| Relaxed Target Kana-CER@1 | 5.006% |
| Sentence Kana-CER | 1.089% |
| Sentence Kana-CER@1 | 1.089% |
| Text CER | 4.226% |
| Text CER@1 | 4.225% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 7,056 | 92.304% | 92.347% | 7.216% | 6.170% | 7.181% | 6.134% |
| Kun’yomi | 6,108 | 95.334% | 95.465% | 4.316% | 3.689% | 4.153% | 3.558% |
| Jōyō appendix readings | 372 | 85.215% | 86.559% | 9.404% | 8.418% | 8.203% | 7.397% |

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
| [Target review](details/target-review.jsonl) | 883 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,935 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 4,726 | Rows with nonzero Text CER |
