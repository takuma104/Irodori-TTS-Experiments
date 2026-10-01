#!/usr/bin/env python3
"""Caption-conditioned samples: base vs base+student vs exported checkpoint.

    PYTHONPATH=Irodori-TTS:scripts uv run --project Irodori-TTS --no-sync python \
        scripts/caption_drift_listen.py ../Irodori-TTS-v4.1-Small-Yomi/model.safetensors \
        --student outputs/yomi_prod/s10_cont/part2/student --output-dir outputs/export_check/caption

The base model with the student installed keeps the original caption path, so
``student`` vs ``exported`` isolates the effect of the shared-backbone drift on
caption conditioning (the text path is identical). Writes
``<caption>_<sentence>_<system>.wav`` and ``summary.json`` with the median F0
and duration of every sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from batch_synth import DEFAULT_HF_CHECKPOINT, load_runtime
from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest
from student_text import install_student

CAPTIONS = [
    "落ち着いた、近い距離感の女性話者",
    "低く渋い声の年配の男性が、ゆっくりと語りかける。",
    "明るく元気な若い女性が、友達に楽しそうに話しかけている。",
    "泣きそうな声で、震えながら話す少女。",
    "早口でまくしたてる、慌てた様子の若い男性。",
    "ささやくように、耳元で優しく話しかける女性。",
    "電話越しのような、こもった音質の女性の声。",
    "怒りを抑えながら、低い声で静かに話す男性。",
]
SENTENCES = [
    "今日は朝から雨が降っていたので、駅まで傘をさして歩きました。",
    "この企画については、来週の会議でもう一度話し合いましょう。",
]


def f0_median(audio: np.ndarray, sample_rate: int) -> float:
    f0, voiced, _ = librosa.pyin(
        audio, fmin=60.0, fmax=600.0, sr=sample_rate, frame_length=2048
    )
    values = f0[voiced & ~np.isnan(f0)]
    return float(np.median(values)) if values.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Exported model.safetensors")
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    systems: dict[str, tuple[str, Path | None]] = {
        "base": (DEFAULT_HF_CHECKPOINT, None),
        "student": (DEFAULT_HF_CHECKPOINT, args.student),
        "exported": (args.checkpoint, None),
    }
    summary: list[dict[str, object]] = []
    for name, (checkpoint, student) in systems.items():
        runtime: InferenceRuntime = load_runtime(checkpoint, precision="fp32")
        if student is not None:
            install_student(runtime.model, student)
        for ci, caption in enumerate(CAPTIONS):
            for si, text in enumerate(SENTENCES):
                result = runtime.synthesize(
                    SamplingRequest(
                        text=text, caption=caption, no_ref=True, seed=args.seed
                    )
                )
                audio = result.audio.float().mean(dim=0).cpu().numpy()
                path = args.output_dir / f"c{ci}_s{si}_{name}.wav"
                sf.write(path, audio, result.sample_rate)
                summary.append(
                    {
                        "system": name,
                        "caption": ci,
                        "sentence": si,
                        "f0_median": round(f0_median(audio, result.sample_rate), 1),
                        "seconds": round(len(audio) / result.sample_rate, 2),
                    }
                )
        del runtime
        torch.cuda.empty_cache()

    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {"captions": CAPTIONS, "sentences": SENTENCES, "samples": summary},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    by = {(s["system"], s["caption"], s["sentence"]): s for s in summary}
    print("caption  " + "  ".join(f"{n:>16s}" for n in systems))
    for ci in range(len(CAPTIONS)):
        cells = []
        for name in systems:
            f0 = [by[(name, ci, si)]["f0_median"] for si in range(len(SENTENCES))]
            cells.append("/".join(f"{v:.0f}" for v in f0) + " Hz")
        print(f"c{ci}       " + "  ".join(f"{c:>16s}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
