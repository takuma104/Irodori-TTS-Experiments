#!/usr/bin/env python3
"""Where does the kana teacher differ from the base, and what did a student fix?

    PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync \
        python scripts/diagnose_distill.py --student outputs/yomi_pilot/s1_projector/student

On rows the base misreads but the kana teacher reads correctly, computes the
conditional velocity of the teacher (kana text), the base (kanji text), and the
student (kanji text) on the same x_t built from the teacher latent, for several
t. Frames are split into the target window (the target's position in the
reading, mapped proportionally onto the latent, plus a margin) and the rest.
Reports the share of the teacher-base difference inside the window and how
much of it the student removed inside and outside the window.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from batch_synth import load_runtime, set_sdpa_backend
from irodori_tts.inference_runtime import SamplingRequest
from irodori_tts.text_normalization import normalize_text
from student_text import StudentTextEncoder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_window(row: dict[str, Any], frames: int, margin: float) -> tuple[int, int]:
    tagged = row["tagged_yomi"]
    start = tagged.index("<")
    end = tagged.index(">") - 1
    total = len(tagged) - 2
    lo = max(0.0, start / total - margin)
    hi = min(1.0, end / total + margin)
    return int(lo * frames), max(int(lo * frames) + 1, int(hi * frames))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, default=Path("outputs/yomi_pilot"))
    parser.add_argument("--num", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--split", default=None, help="Only rows of this split.")
    args = parser.parse_args()

    rows = {
        r["key"]: r for r in read_jsonl(args.pilot_dir / "teacher" / "dataset.jsonl")
    }
    manifest = {
        r["key"]: r for r in read_jsonl(args.pilot_dir / "teacher" / "manifest.jsonl")
    }
    base_ok = {
        r["key"]: r["target_exact"]
        for r in read_jsonl(
            args.pilot_dir / "base_kanji" / "results" / "details" / "all.jsonl"
        )
    }
    teacher_ok = {
        r["key"]: r["target_exact"]
        for r in read_jsonl(
            args.pilot_dir / "teacher" / "results" / "details" / "all.jsonl"
        )
    }
    hard = sorted(
        k
        for k in base_ok
        if not base_ok[k]
        and teacher_ok.get(k)
        and (args.split is None or rows[k].get("split") == args.split)
    )
    keys = random.Random(0).sample(hard, min(args.num, len(hard)))

    set_sdpa_backend("efficient")
    runtime = load_runtime(
        precision="bf16", text_precision="fp32", codec_precision="fp32"
    )
    model = runtime.model
    student = StudentTextEncoder.load(model, args.student).to(runtime.model_device)
    dit = model.in_proj.weight.dtype
    device = runtime.model_device
    torch.manual_seed(0)

    def encode(text: str, ref: str | None, use_student: bool) -> tuple[Any, ...]:
        ids, mask = runtime.tokenizer.batch_encode(
            [normalize_text(text).strip()], max_length=int(runtime.default_text_max_len)
        )
        ids, mask = ids.to(device), mask.to(device)
        ref_latent, ref_mask = runtime._load_reference_latent(
            req=SamplingRequest(text="", ref_wav=ref, no_ref=ref is None),
            batch_size=1,
            messages=[],
        )
        cap_ids, cap_mask = runtime.caption_tokenizer.batch_encode(
            [""], max_length=int(runtime.default_caption_max_len)
        )
        cap_mask.zero_()
        states = model.encode_conditions(
            text_input_ids=ids,
            text_mask=mask,
            ref_latent=ref_latent.to(dit),
            ref_mask=ref_mask,
            caption_input_ids=cap_ids.to(device),
            caption_mask=cap_mask.to(device),
        )
        text_state = states[0]
        if use_student:
            text_state = student.encode(model.pretrained_text_backbone, ids, mask)
        return (
            text_state.to(dit),
            states[1],
            states[2],
            states[3],
            states[4].to(dit),
            states[5],
        )

    ts = [0.9, 0.7, 0.5, 0.3]
    sums = {t: {"in_tb": 0.0, "out_tb": 0.0, "in_ts": 0.0, "out_ts": 0.0} for t in ts}
    frames_in = frames_out = 0
    with torch.no_grad():
        for key in keys:
            row, meta = rows[key], manifest[key]
            x0 = torch.load(args.pilot_dir / "teacher" / "latents" / f"{key}.pt").to(
                device
            )[None]
            frames = x0.shape[1]
            lo, hi = target_window(row, frames, args.margin)
            inside = torch.zeros(frames, dtype=torch.bool, device=device)
            inside[lo:hi] = True
            frames_in += int(inside.sum())
            frames_out += int((~inside).sum())
            cond = {
                "teacher": encode(meta["text"], meta["ref_wav"], False),
                "base": encode(meta["kanji_text"], meta["ref_wav"], False),
                "student": encode(meta["kanji_text"], meta["ref_wav"], True),
            }
            noise = torch.randn_like(x0)
            for t in ts:
                x_t = ((1 - t) * x0 + t * noise).to(dit)
                tt = torch.full((1,), t, device=device, dtype=dit)
                v = {
                    name: model.forward_with_encoded_conditions(
                        x_t=x_t,
                        t=tt,
                        text_state=c[0],
                        text_mask=c[1],
                        speaker_state=c[2],
                        speaker_mask=c[3],
                        caption_state=c[4],
                        caption_mask=c[5],
                    ).float()[0]
                    for name, c in cond.items()
                }
                d_tb = ((v["teacher"] - v["base"]) ** 2).mean(dim=-1)
                d_ts = ((v["teacher"] - v["student"]) ** 2).mean(dim=-1)
                sums[t]["in_tb"] += float(d_tb[inside].sum())
                sums[t]["out_tb"] += float(d_tb[~inside].sum())
                sums[t]["in_ts"] += float(d_ts[inside].sum())
                sums[t]["out_ts"] += float(d_ts[~inside].sum())

    print(f"rows={len(keys)} frames in window={frames_in} outside={frames_out}")
    print(
        "t    share_in_window  per-frame(in/out)  student_removed_in  student_removed_out"
    )
    for t in ts:
        s = sums[t]
        share = s["in_tb"] / (s["in_tb"] + s["out_tb"])
        ratio = (s["in_tb"] / frames_in) / (s["out_tb"] / frames_out)
        print(
            f"{t:.1f}  {share:.1%}            {ratio:.1f}x               "
            f"{1 - s['in_ts'] / s['in_tb']:+.1%}             {1 - s['out_ts'] / s['out_tb']:+.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
