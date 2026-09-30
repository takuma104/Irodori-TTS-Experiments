# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/jkyb/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/jkyb/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/jkyb/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 13,536 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 13,536 / 13,536 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 94.880% (12,843 / 13,536) |
| Relaxed Accuracy | 94.976% (12,856 / 13,536) |
| Target Kana-CER | 4.488% |
| Target Kana-CER@1 | 3.968% |
| Relaxed Target Kana-CER | 4.388% |
| Relaxed Target Kana-CER@1 | 3.881% |
| Sentence Kana-CER | 0.810% |
| Sentence Kana-CER@1 | 0.810% |
| Text CER | 3.998% |
| Text CER@1 | 3.996% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 7,056 | 94.090% | 94.118% | 5.284% | 4.710% | 5.263% | 4.689% |
| Kun’yomi | 6,108 | 96.284% | 96.398% | 3.292% | 2.875% | 3.161% | 2.760% |
| Jōyō appendix readings | 372 | 86.828% | 87.903% | 9.028% | 7.863% | 7.935% | 6.949% |

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
| [Target review](details/target-review.jsonl) | 693 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,574 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 4,556 | Rows with nonzero Text CER |
