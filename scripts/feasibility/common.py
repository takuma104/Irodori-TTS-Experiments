"""Shared helpers for the size-distillation feasibility measurements.

Run every script in this directory inside the Irodori-TTS environment:

    PYTHONPATH=third-party/Irodori-TTS:scripts/feasibility \
        third-party/Irodori-TTS/.venv/bin/python scripts/feasibility/<script>.py

JSUT BASIC5000 provides the sentences and the two reference voices. Extract
``basic5000/wav/BASIC5000_000{1,2}.wav`` and ``basic5000/transcript_utf8.txt``
from ``jsut_ver1.1.zip`` into ``data/jsut/``.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import irodori_tts.inference_runtime as ir
import torch
from huggingface_hub import snapshot_download
from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
JSUT_DIR = REPO_ROOT / "data" / "jsut"
OUTPUT_DIR = REPO_ROOT / "outputs" / "feasibility"
REPOS = {
    "RF": "Aratako/Irodori-TTS-v4.1-Small",
    "MF": "Aratako/Irodori-TTS-v4.1-Small-MF",
}
MIB = 1024**2


def checkpoint_path(name: str) -> Path:
    """``model.safetensors`` for ``RF`` / ``MF`` (downloaded if missing), or a local checkpoint dir/file."""
    local = Path(name)
    if local.is_file():
        return local
    if (local / "model.safetensors").is_file():
        return local / "model.safetensors"
    snapshot = snapshot_download(REPOS[name], allow_patterns=["model.safetensors", "tokenizer/*"])
    return Path(snapshot) / "model.safetensors"


def reference_wav(index: int) -> str:
    return str(JSUT_DIR / f"BASIC5000_{index:04d}.wav")


def jsut_sentences(seed: int) -> list[str]:
    lines = (JSUT_DIR / "transcript_utf8.txt").read_text(encoding="utf-8").splitlines()
    texts = [line.split(":", 1)[1].strip() for line in lines if ":" in line]
    random.Random(seed).shuffle(texts)
    return texts


def load_model(name: str, device: str = "cuda") -> tuple[TextToLatentRFDiT, PretrainedTextTokenizer]:
    """Build the model from checkpoint metadata exactly as the inference runtime does."""
    path = checkpoint_path(name)
    state, cfg_dict, _, text_encoder_cfg = ir._load_checkpoint_for_inference(path)
    cfg = merge_dataclass_overrides(ModelConfig(), cfg_dict, section="checkpoint model_config")
    model = TextToLatentRFDiT(
        cfg,
        pretrained_backbone_config=text_encoder_cfg,
        load_pretrained_backbone_weights=False,
    )
    model.load_state_dict(state, assign=True)
    model = model.to(device).eval()
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=str(path.parent / "tokenizer"), add_bos=True, local_files_only=True
    )
    return model, tokenizer


def load_runtime(name: str, model_precision: str, codec_precision: str = "fp32") -> InferenceRuntime:
    key = RuntimeKey(
        checkpoint=str(checkpoint_path(name)),
        model_device="cuda",
        model_precision=model_precision,
        codec_device="cuda",
        codec_precision=codec_precision,
    )
    return InferenceRuntime.from_key(key)


class SamplerCapture:
    """Records the keyword arguments and output latent of every sampler call made by the runtime."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._originals: dict[str, Callable[..., torch.Tensor]] = {}

    def __enter__(self) -> SamplerCapture:
        for attr in ("sample_euler_rf_cfg", "sample_euler_meanflow"):
            original = getattr(ir, attr)
            self._originals[attr] = original
            setattr(ir, attr, self._wrap(original))
        return self

    def __exit__(self, *exc: object) -> None:
        for attr, original in self._originals.items():
            setattr(ir, attr, original)

    def _wrap(self, fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def inner(**kwargs: Any) -> torch.Tensor:
            z = fn(**kwargs)
            record = {key: value for key, value in kwargs.items() if key != "model"}
            self.calls.append({**record, "z": z.detach().clone()})
            return z

        return inner


def synthesize_captured(
    runtime: InferenceRuntime, texts: list[str]
) -> list[dict[str, Any]]:
    """Synthesize ``texts`` (alternating the two JSUT voices) and return the sampler inputs/outputs."""
    with SamplerCapture() as capture:
        for i, text in enumerate(texts):
            runtime.synthesize(
                SamplingRequest(text=text, ref_wav=reference_wav(1 + i % 2), seed=i)
            )
    return capture.calls
