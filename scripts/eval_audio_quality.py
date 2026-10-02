#!/usr/bin/env python3
"""Speaker similarity and predicted naturalness of synthesized audio (plan Phase 0.4).

    PYTHONPATH=third-party/Irodori-TTS:scripts third-party/Irodori-TTS/.venv/bin/python \
        scripts/eval_audio_quality.py outputs/jkyb/mf_teacher_jsut --ref-wav data/jsut/BASIC5000_0001.wav \
        --keys-from data/eval/jkyb_dev.jsonl

For every ``<run>/audio/<key>.wav`` (optionally restricted to the keys of a
JSONL file) computes:

- ``sim``: cosine similarity between WavLM-Base-Plus-SV x-vectors of the
  utterance and of the reference voice;
- ``utmos``: UTMOS22 (strong) predicted MOS, via SpeechMOS;
- ``seconds``: duration.

Writes ``<run>/quality.jsonl`` (per file) and ``<run>/quality_summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, WavLMForXVector

SV_MODEL = "microsoft/wavlm-base-plus-sv"
UTMOS_HUB = ("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")


def load_16k(path: Path) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(wav.T).mean(dim=0)
    return torchaudio.functional.resample(audio, sr, 16000) if sr != 16000 else audio


class QualityScorer:
    def __init__(self, device: str) -> None:
        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        self.sv = WavLMForXVector.from_pretrained(SV_MODEL).to(device).eval()
        self.utmos = torch.hub.load(*UTMOS_HUB, trust_repo=True).to(device).eval()

    @torch.inference_mode()
    def embedding(self, audio: torch.Tensor) -> torch.Tensor:
        inputs = self.extractor(audio.numpy(), sampling_rate=16000, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(self.sv(**inputs).embeddings[0], dim=-1)

    @torch.inference_mode()
    def mos(self, audio: torch.Tensor) -> float:
        return float(self.utmos(audio[None].to(self.device), 16000)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--ref-wav", type=Path, required=True)
    parser.add_argument("--keys-from", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    paths = sorted((args.run / "audio").glob("*.wav"))
    if args.keys_from is not None:
        keys = {json.loads(line)["key"] for line in args.keys_from.read_text(encoding="utf-8").splitlines() if line}
        paths = [p for p in paths if p.stem in keys]
    scorer = QualityScorer(args.device)
    reference = scorer.embedding(load_16k(args.ref_wav))
    rows = []
    for path in paths:
        audio = load_16k(path)
        rows.append(
            {
                "key": path.stem,
                "sim": float(scorer.embedding(audio) @ reference),
                "utmos": scorer.mos(audio),
                "seconds": audio.numel() / 16000,
            }
        )
    with (args.run / "quality.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "files": len(rows),
        "keys_from": str(args.keys_from) if args.keys_from else None,
        "sim_mean": sum(r["sim"] for r in rows) / len(rows),
        "utmos_mean": sum(r["utmos"] for r in rows) / len(rows),
        "seconds_mean": sum(r["seconds"] for r in rows) / len(rows),
    }
    (args.run / "quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
