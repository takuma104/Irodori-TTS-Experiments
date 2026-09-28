#!/usr/bin/env python3
"""Generate teacher latents and audio for the yomi dataset (plan §4.5).

    PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync \
        python scripts/generate_teacher_latents.py data/yomi/rows_pilot.jsonl \
        --output-dir outputs/yomi_pilot/teacher

Each row gets a reference speaker chosen by a hash of its key (JVS speakers other
than jvs001, which is kept for evaluation, or no reference) and a per-key seed.
The synthesized text is chosen per row:

- ``--text-mode teacher`` (default): ``--kana-field`` for target rows, ``text``
  (kanji) for contrast rows. These latents are the distillation targets.
- ``--text-mode kanji``: always ``text``, for hard mining with the current model.

Writes ``audio/<key>.wav``, ``latents/<key>.pt`` (untrimmed, float32, frames x
latent_dim), ``manifest.jsonl``, and ``dataset.jsonl`` (the input rows that were
synthesized, for ``jkyb-eval tts --dataset``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from batch_synth import BatchSynthesizer, SynthItem, load_runtime, set_sdpa_backend
from irodori_tts.inference_runtime import save_wav

EVAL_SPEAKER = "jvs001"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key_hash(key: str, salt: str) -> int:
    return int.from_bytes(hashlib.md5(f"{salt}:{key}".encode()).digest()[:4], "little")


def speaker_pool(jvs_dir: Path) -> list[str]:
    return sorted(
        str(path)
        for path in jvs_dir.glob(
            "jvs*/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"
        )
        if path.parts[-4] != EVAL_SPEAKER
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-mode", choices=["teacher", "kanji"], default="teacher")
    parser.add_argument(
        "--kana-field",
        default="kana_text_kata",
        help="Kana-substituted text for target rows (M0 chose katakana).",
    )
    parser.add_argument("--jvs-dir", type=Path, default=Path("data/jvs_ver1"))
    parser.add_argument("--noref-ratio", type=float, default=0.2)
    parser.add_argument("--split", default=None, help="Only rows of this split.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-frames", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    if args.split is not None:
        rows = [r for r in rows if r.get("split") == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]
    speakers = speaker_pool(args.jvs_dir)
    if not speakers:
        raise SystemExit(f"No reference speakers found under {args.jvs_dir}")

    out = args.output_dir
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    groups: dict[str | None, list[tuple[dict[str, Any], SynthItem]]] = defaultdict(list)
    for row in rows:
        if (out / "latents" / f"{row['key']}.pt").is_file():
            continue
        use_kana = args.text_mode == "teacher" and row.get("role") == "target"
        text = row[args.kana_field] if use_kana else row["text"]
        h = key_hash(row["key"], "speaker")
        ref = (
            None
            if (h % 1000) < args.noref_ratio * 1000
            else speakers[h % len(speakers)]
        )
        item = SynthItem(
            key=row["key"], text=text, seed=key_hash(row["key"], "seed") >> 1
        )
        groups[ref].append((row, item))
    todo = sum(len(v) for v in groups.values())
    print(f"rows={len(rows)} todo={todo} speakers={len(groups)}", flush=True)

    with (out / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    set_sdpa_backend("efficient")
    runtime = load_runtime(
        precision=args.precision, text_precision="fp32", codec_precision="fp32"
    )
    start = time.perf_counter()
    done = 0
    with (out / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
        for ref, members in sorted(groups.items(), key=lambda kv: kv[0] or ""):
            synth = BatchSynthesizer(runtime, ref_wav=ref)
            rows_by_key = {row["key"]: row for row, _ in members}
            items = [item for _, item in members]
            text_by_key = {item.key: item.text for item in items}
            for output in synth.synthesize(
                items,
                max_batch_size=args.batch_size,
                max_batch_frames=args.max_batch_frames,
            ):
                save_wav(
                    out / "audio" / f"{output.key}.wav",
                    output.audio,
                    output.sample_rate,
                )
                torch.save(
                    output.latent.float().clone(), out / "latents" / f"{output.key}.pt"
                )
                row = rows_by_key[output.key]
                manifest.write(
                    json.dumps(
                        {
                            "key": output.key,
                            "text": text_by_key[output.key],
                            "kanji_text": row["text"],
                            "ref_wav": ref,
                            "frames": int(output.latent.shape[0]),
                            "role": row.get("role"),
                            "bucket": row.get("bucket"),
                            "split": row.get("split"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                done += 1
            manifest.flush()
            elapsed = time.perf_counter() - start
            print(
                f"{done}/{todo} elapsed={elapsed:.0f}s rate={done / elapsed:.2f}/s",
                flush=True,
            )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
