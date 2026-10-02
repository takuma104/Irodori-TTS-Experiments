#!/usr/bin/env python3
"""Reference-voice pool for distillation: DACVAE latents of real speech clips.

    PYTHONPATH=third-party/Irodori-TTS:scripts third-party/Irodori-TTS/.venv/bin/python \
        scripts/build_ref_pool.py --output data/refs/real_pool.pt

Reads clips straight from the corpus zips under /mnt/datasets/speech/free_distribute,
encodes them like the inference runtime encodes ``--ref-wav`` (loudness
normalized to -16 dB, peak-safe), and stores one file with the float16 latents
and per-clip metadata (corpus, speaker, source member, seconds). Each corpus is
tagged so that it can be dropped later if its license does not allow the
release (plan Phase 0.5). JSUT is excluded: its speaker is the evaluation
reference voice.
"""

from __future__ import annotations

import argparse
import io
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from irodori_tts.codec import DACVAECodec

ROOT = Path("/mnt/datasets/speech/free_distribute")


@dataclass(frozen=True)
class Source:
    corpus: str
    zip_path: Path
    member: re.Pattern[str]
    speaker: str | None  # fixed speaker id, or None to take the first regex group
    limit: int  # clips per source (random subset)


SOURCES = [
    Source("jvnv", ROOT / "jvnv/origin/jvnv_ver1.zip", re.compile(r"jvnv_v1/(F1|F2|M1|M2)/.+\.wav$"), None, 2000),
    Source("hifi_captain", ROOT / "hi-fi-captain/origin/hfc_ja-JP_F.zip",
           re.compile(r"ja-JP/female/wav/train_parallel/.+\.wav$"), "hfc_ja_female", 400),
    Source("hifi_captain", ROOT / "hi-fi-captain/origin/hfc_ja-JP_M.zip",
           re.compile(r"ja-JP/male/wav/train_parallel/.+\.wav$"), "hfc_ja_male", 400),
    Source("amitaro", ROOT / "amitaro/origin/ITAcorpus_amitaro_2.1.zip", re.compile(r".+\.wav$"), "amitaro", 300),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/refs/real_pool.pt"))
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    codec = DACVAECodec.load(
        repo_id="Aratako/Semantic-DACVAE-Japanese-32dim", device=args.device, dtype=torch.float32
    )
    rng = random.Random(args.seed)
    latents: list[torch.Tensor] = []
    meta: list[dict[str, str | float]] = []
    for source in SOURCES:
        with zipfile.ZipFile(source.zip_path) as archive:
            names = sorted(n for n in archive.namelist() if source.member.search(n))
            rng.shuffle(names)
            taken = 0
            for name in names:
                if taken >= source.limit:
                    break
                wav, sr = sf.read(io.BytesIO(archive.read(name)), dtype="float32", always_2d=True)
                seconds = wav.shape[0] / sr
                if not args.min_seconds <= seconds <= args.max_seconds:
                    continue
                audio = torch.from_numpy(wav.T).mean(dim=0, keepdim=True)  # mono (1, T)
                with torch.inference_mode():
                    latent = codec.encode_waveform(
                        audio.unsqueeze(0), sample_rate=int(sr), normalize_db=-16.0, ensure_max=True
                    )
                latents.append(latent[0].to("cpu", torch.float16))
                match = source.member.search(name)
                speaker = source.speaker or f"{source.corpus}_{match.group(1)}"
                meta.append({"corpus": source.corpus, "speaker": speaker, "member": name, "seconds": seconds})
                taken += 1
        print(f"{source.corpus} {source.zip_path.name}: {taken} clips", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents, "meta": meta}, args.output)
    speakers = sorted({str(m["speaker"]) for m in meta})
    print(f"wrote {args.output}: {len(latents)} clips, {len(speakers)} speakers: {speakers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
