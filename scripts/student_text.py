"""Trainable text-path encoder for kana-substitution distillation (plan §5.1-5.2).

Irodori-TTS v4.1 shares one ModernBERT between the reading text and the caption,
with separate projectors. ``StudentTextEncoder`` copies the text projector and
``text_norm`` and, when BERT layers are trained, the backbone too, so the
caption path keeps the original weights. ``install_student`` swaps it into a
loaded model for inference without changing Irodori's code:
``encode_conditions`` calls ``model.text_encoder(shared_backbone, ids, mask)``
and then ``model.text_norm``, and both are replaced.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import torch
from irodori_tts.model import TextToLatentRFDiT
from safetensors.torch import load_file, save_file
from torch import nn

SCOPE_PATTERN = re.compile(r"^(projector|all|top(\d+))$")


class StudentTextEncoder(nn.Module):
    def __init__(self, model: TextToLatentRFDiT, scope: str) -> None:
        super().__init__()
        match = SCOPE_PATTERN.match(scope)
        if match is None:
            raise ValueError(f"scope must be projector, all, or top<N>, got {scope!r}")
        if model.pretrained_text_backbone is None:
            raise ValueError("The model has no pretrained text backbone.")
        self.scope = scope
        self.backbone = (
            None
            if scope == "projector"
            else copy.deepcopy(model.pretrained_text_backbone)
        )
        self.projector = copy.deepcopy(model.text_encoder)
        self.norm = copy.deepcopy(model.text_norm)
        self.float()
        self.requires_grad_(False)
        for parameter in self.trainable_parameters():
            parameter.requires_grad_(True)

    def trainable_names(self) -> list[str]:
        names = [
            n
            for n, _ in self.named_parameters()
            if n.startswith(("projector.", "norm."))
        ]
        if self.backbone is None:
            return names
        layers = self.backbone.backbone.layers
        match = SCOPE_PATTERN.match(self.scope)
        first = 0 if self.scope == "all" else len(layers) - int(match.group(2))
        prefixes = [f"backbone.backbone.layers.{i}." for i in range(first, len(layers))]
        prefixes.append("backbone.backbone.final_norm.")
        if self.scope == "all":
            prefixes.append("backbone.backbone.embeddings.")
        names += [
            n for n, _ in self.named_parameters() if n.startswith(tuple(prefixes))
        ]
        return names

    def trainable_parameters(self) -> list[nn.Parameter]:
        wanted = set(self.trainable_names())
        return [p for n, p in self.named_parameters() if n in wanted]

    def encode(
        self, shared_backbone: nn.Module, input_ids: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        backbone = self.backbone if self.backbone is not None else shared_backbone
        return self.norm(self.projector(backbone, input_ids, mask))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        wanted = set(self.trainable_names())
        tensors = {
            n: p.detach().float().cpu().contiguous()
            for n, p in self.named_parameters()
            if n in wanted
        }
        save_file(tensors, str(path / "student.safetensors"))
        (path / "student_config.json").write_text(
            json.dumps({"scope": self.scope}, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, model: TextToLatentRFDiT, path: Path) -> StudentTextEncoder:
        config = json.loads((path / "student_config.json").read_text(encoding="utf-8"))
        student = cls(model, config["scope"])
        state = load_file(str(path / "student.safetensors"))
        missing = set(student.trainable_names()) - set(state)
        if missing:
            raise ValueError(f"Student checkpoint is missing {sorted(missing)[:5]}")
        student.load_state_dict(state, strict=False)
        return student


class _ProjectorOverride(nn.Module):
    """Stands in for ``model.text_encoder`` and routes to the student's backbone."""

    def __init__(self, student: StudentTextEncoder) -> None:
        super().__init__()
        self.student = student

    def forward(
        self, shared_backbone: nn.Module, input_ids: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        backbone = (
            self.student.backbone
            if self.student.backbone is not None
            else shared_backbone
        )
        return self.student.projector(backbone, input_ids, mask)


def install_student(model: TextToLatentRFDiT, path: Path) -> StudentTextEncoder:
    """Load a trained student and make ``model`` use it for the text condition."""
    # Match the text path, which may be fp32 while the DiT runs in bf16.
    dtype = model.text_norm.weight.dtype
    device = model.text_norm.weight.device
    student = StudentTextEncoder.load(model, path).to(device=device, dtype=dtype)
    student.requires_grad_(False)
    student.eval()
    model.text_encoder = _ProjectorOverride(student)
    model.text_norm = student.norm
    return student
