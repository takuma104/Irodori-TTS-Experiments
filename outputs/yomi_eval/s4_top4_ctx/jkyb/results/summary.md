# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s4_top4_ctx/jkyb/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s4_top4_ctx/jkyb/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s4_top4_ctx/jkyb/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 13,536 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 13,536 / 13,536 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 94.134% (12,742 / 13,536) |
| Relaxed Accuracy | 94.230% (12,755 / 13,536) |
| Target Kana-CER | 5.385% |
| Target Kana-CER@1 | 4.631% |
| Relaxed Target Kana-CER | 5.280% |
| Relaxed Target Kana-CER@1 | 4.539% |
| Sentence Kana-CER | 0.918% |
| Sentence Kana-CER@1 | 0.918% |
| Text CER | 3.967% |
| Text CER@1 | 3.965% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 7,056 | 93.112% | 93.155% | 6.493% | 5.563% | 6.458% | 5.527% |
| Kun’yomi | 6,108 | 95.842% | 95.956% | 3.810% | 3.286% | 3.679% | 3.171% |
| Jōyō appendix readings | 372 | 85.484% | 86.290% | 10.224% | 9.059% | 9.238% | 8.253% |

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
| [Target review](details/target-review.jsonl) | 794 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,686 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 4,539 | Rows with nonzero Text CER |
