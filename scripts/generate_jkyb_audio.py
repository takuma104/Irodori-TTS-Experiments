#!/usr/bin/env python3
"""Synthesize JKYB-Parakeet sentences with Irodori-TTS.

Run inside the Irodori-TTS environment, e.g.:

    uv run --project Irodori-TTS --no-sync python scripts/generate_jkyb_audio.py \
        --output-dir outputs/jkyb/irodori-v4.1-small/audio

Each dataset row's ``text`` is synthesized and saved as ``<key>.wav``.
Existing files are skipped so interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    save_wav,
)

DATASET_REPO = "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet"
DATASET_FILENAME = "data/common_kanji_source.jsonl"


def load_rows(dataset: Path | None) -> list[dict[str, str]]:
    path = (
        dataset
        if dataset is not None
        else Path(
            hf_hub_download(
                repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILENAME
            )
        )
    )
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw = json.loads(line)
                rows.append({"key": raw["key"], "text": raw["text"]})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--ref-wav", default=None, help="Omit to run with --no-ref.")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.dataset)[args.shard :: args.num_shards]
    if args.limit is not None:
        rows = rows[: args.limit]
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir.parent / f"generation_config_shard{args.shard}.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    checkpoint = download_hf_checkpoint(args.hf_checkpoint)
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(checkpoint),
            model_device=args.device,
            model_precision=args.precision,
            codec_device=args.device,
            codec_precision=args.precision,
        )
    )

    todo = [row for row in rows if not (out_dir / f"{row['key']}.wav").is_file()]
    print(f"rows={len(rows)} todo={len(todo)}", flush=True)
    failures: list[dict[str, str]] = []
    start = time.perf_counter()
    for index, row in enumerate(todo, start=1):
        try:
            result = runtime.synthesize(
                SamplingRequest(
                    text=row["text"],
                    caption=args.caption,
                    ref_wav=args.ref_wav,
                    no_ref=args.ref_wav is None,
                    num_steps=args.num_steps,
                    seed=args.seed,
                    cfg_scale_speaker=5.0 if args.ref_wav is not None else 0.0,
                ),
                log_fn=None,
            )
            save_wav(out_dir / f"{row['key']}.wav", result.audio, result.sample_rate)
        except Exception as error:  # noqa: BLE001 - keep going over the whole corpus
            failures.append({"key": row["key"], "error": repr(error)})
            print(f"error key={row['key']}: {error!r}", file=sys.stderr, flush=True)
        if index % 100 == 0 or index == len(todo):
            elapsed = time.perf_counter() - start
            eta = elapsed / index * (len(todo) - index)
            print(
                f"progress {index}/{len(todo)} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    if failures:
        failure_path = out_dir.parent / f"generation_failures_shard{args.shard}.jsonl"
        with failure_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    print(f"done failures={len(failures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
