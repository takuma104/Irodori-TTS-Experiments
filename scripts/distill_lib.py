"""Checkpoint I/O and student construction shared by the size-distillation scripts.

Every student is a complete ``TextToLatentRFDiT`` whose shape is described by the
two safetensors metadata entries the stock inference runtime reads:

- ``config_json``: the flat ``ModelConfig`` plus the inference keys
  (``max_text_len``, ``max_caption_len``, ``ref_max_seconds``);
- ``text_encoder_config_json``: the Hugging Face config of the text backbone.

``save_checkpoint`` writes both together with the weights and the bundled
tokenizer, so the output directory loads with an unmodified
``infer.py --checkpoint <dir>/model.safetensors``.
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.model import TextToLatentRFDiT
from safetensors import safe_open
from safetensors.torch import load_file, save_file

TEACHER_REPOS = {
    "RF": "Aratako/Irodori-TTS-v4.1-Small",
    "MF": "Aratako/Irodori-TTS-v4.1-Small-MF",
}
CONFIG_META_KEY = "config_json"
TEXT_ENCODER_CONFIG_META_KEY = "text_encoder_config_json"
INFERENCE_CONFIG_KEYS = {"max_text_len", "max_caption_len", "fixed_target_latent_steps", "ref_max_seconds"}
BACKBONE_PREFIX = "pretrained_text_backbone.backbone."
TEXT_PATH_PREFIXES = (
    "pretrained_text_backbone.",
    "text_encoder.",
    "caption_encoder.",
    "text_norm.",
    "caption_norm.",
)


@dataclass
class Checkpoint:
    path: Path
    flat_config: dict[str, Any]
    text_encoder_config: dict[str, Any]
    state: dict[str, torch.Tensor]

    @property
    def tokenizer_dir(self) -> Path:
        return self.path.parent / "tokenizer"

    @property
    def model_config(self) -> ModelConfig:
        return model_config_from_flat(self.flat_config)


def resolve_checkpoint_path(source: str | Path) -> Path:
    """``RF`` / ``MF``, a Hugging Face repo id, a checkpoint directory, or a ``model.safetensors``."""
    source = str(source)
    repo = TEACHER_REPOS.get(source, source)
    local = Path(repo)
    if local.is_file():
        return local
    if (local / "model.safetensors").is_file():
        return local / "model.safetensors"
    snapshot = snapshot_download(repo, allow_patterns=["model.safetensors", "tokenizer/*"])
    return Path(snapshot) / "model.safetensors"


def load_checkpoint(source: str | Path) -> Checkpoint:
    path = resolve_checkpoint_path(source)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    return Checkpoint(
        path=path,
        flat_config=json.loads(metadata[CONFIG_META_KEY]),
        text_encoder_config=json.loads(metadata[TEXT_ENCODER_CONFIG_META_KEY]),
        state=load_file(str(path), device="cpu"),
    )


def model_config_from_flat(flat_config: dict[str, Any]) -> ModelConfig:
    model_keys = {k: v for k, v in flat_config.items() if k not in INFERENCE_CONFIG_KEYS}
    return merge_dataclass_overrides(ModelConfig(), model_keys, section="checkpoint model_config")


def build_model(
    flat_config: dict[str, Any],
    text_encoder_config: dict[str, Any],
    state: dict[str, torch.Tensor] | None = None,
) -> TextToLatentRFDiT:
    """Instantiate a model from checkpoint metadata (as the inference runtime does) on CPU, fp32."""
    model = TextToLatentRFDiT(
        model_config_from_flat(flat_config),
        pretrained_backbone_config=copy.deepcopy(text_encoder_config),
        load_pretrained_backbone_weights=False,
    )
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model.float()


def save_checkpoint(
    model: TextToLatentRFDiT,
    *,
    flat_config: dict[str, Any],
    text_encoder_config: dict[str, Any],
    output_dir: Path,
    tokenizer_dir: Path,
) -> Path:
    """Write ``model.safetensors`` (fp32, stock metadata layout) and the bundled tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()}
    metadata = {
        CONFIG_META_KEY: json.dumps(flat_config, ensure_ascii=False, separators=(",", ":")),
        TEXT_ENCODER_CONFIG_META_KEY: json.dumps(
            text_encoder_config, ensure_ascii=False, separators=(",", ":")
        ),
    }
    path = output_dir / "model.safetensors"
    save_file(state, str(path), metadata=metadata)
    if tokenizer_dir.resolve() != (output_dir / "tokenizer").resolve():
        shutil.copytree(tokenizer_dir, output_dir / "tokenizer", dirs_exist_ok=True)
    return path


def count_parameters(state: dict[str, torch.Tensor], prefixes: tuple[str, ...] = ("",)) -> int:
    return sum(t.numel() for k, t in state.items() if k.startswith(prefixes))


# --- text backbone students -------------------------------------------------------------


def pruned_backbone_config(text_encoder_config: dict[str, Any], keep: list[int]) -> dict[str, Any]:
    """HF config of the backbone restricted to the ``keep`` layers (attention types preserved)."""
    if not keep or keep[0] != 0 or sorted(set(keep)) != keep:
        raise ValueError("keep must be strictly increasing and start with layer 0 (no attn_norm).")
    config = copy.deepcopy(text_encoder_config)
    config["layer_types"] = [config["layer_types"][i] for i in keep]
    config["num_hidden_layers"] = len(keep)
    return config


def pruned_backbone_state(
    state: dict[str, torch.Tensor], keep: list[int]
) -> dict[str, torch.Tensor]:
    """Rename ``backbone.layers.{keep[i]}.*`` to ``backbone.layers.{i}.*`` and drop the rest."""
    new_index = {old: new for new, old in enumerate(keep)}
    layer_prefix = BACKBONE_PREFIX + "layers."
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if not key.startswith(layer_prefix):
            out[key] = tensor
            continue
        index, rest = key[len(layer_prefix) :].split(".", 1)
        if int(index) in new_index:
            out[f"{layer_prefix}{new_index[int(index)]}.{rest}"] = tensor
    return out


def pretrained_backbone(repo_id: str, revision: str | None) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """HF config dict and ``pretrained_text_backbone.*`` state of a pretrained encoder."""
    from irodori_tts.model import PretrainedTextBackbone

    backbone = PretrainedTextBackbone(repo_id, revision=revision, load_pretrained_weights=True)
    state = {f"pretrained_text_backbone.{k}": v for k, v in backbone.state_dict().items()}
    return backbone.config_dict, state
