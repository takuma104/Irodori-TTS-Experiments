"""On-the-fly conditions and teacher MeanFlow rollouts for DiT distillation (plan Phase 2).

A training example is a condition triple (reading text, reference latent or
none, caption or none). ``encode_batch`` turns a list of them into the
teacher's encoded conditions and the latent lengths its duration predictor
predicts, exactly as the inference runtime does for single requests;
``meanflow_rollout`` then runs the MeanFlow sampler and records every state
x_t, its (t, delta) and the model output u at that state.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from irodori_tts.duration import build_duration_features
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.text_normalization import normalize_text
from irodori_tts.tokenizer import PretrainedTextTokenizer

FRAMES_PER_SECOND = 25.0  # DACVAE: 48 kHz / hop 1920
MAX_TEXT_LEN = 256
MAX_CAPTION_LEN = 512


@dataclass
class ConditionItem:
    text: str
    ref: torch.Tensor | None  # (frames, latent_dim) or None for no reference
    caption: str | None


@dataclass
class EncodedBatch:
    text_state: torch.Tensor
    text_mask: torch.Tensor
    speaker_state: torch.Tensor | None
    speaker_mask: torch.Tensor | None
    caption_state: torch.Tensor | None
    caption_mask: torch.Tensor | None
    frames: list[int]  # latent length per item (predicted by the encoding model unless given)
    latent_mask: torch.Tensor  # (B, max(frames))
    log_frames: torch.Tensor | None = None  # the encoding model's predicted log1p(frames)

    def conditions(self) -> dict[str, torch.Tensor | None]:
        return {
            "text_state": self.text_state,
            "text_mask": self.text_mask,
            "speaker_state": self.speaker_state,
            "speaker_mask": self.speaker_mask,
            "caption_state": self.caption_state,
            "caption_mask": self.caption_mask,
        }

    def repeat(self, times: int) -> EncodedBatch:
        def rep(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.repeat(times, *([1] * (value.ndim - 1)))

        return EncodedBatch(
            text_state=rep(self.text_state),
            text_mask=rep(self.text_mask),
            speaker_state=rep(self.speaker_state),
            speaker_mask=rep(self.speaker_mask),
            caption_state=rep(self.caption_state),
            caption_mask=rep(self.caption_mask),
            frames=self.frames * times,
            latent_mask=rep(self.latent_mask),
            log_frames=rep(self.log_frames),
        )

    def select(self, index: torch.Tensor) -> EncodedBatch:
        def take(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value[index]

        return EncodedBatch(
            text_state=take(self.text_state),
            text_mask=take(self.text_mask),
            speaker_state=take(self.speaker_state),
            speaker_mask=take(self.speaker_mask),
            caption_state=take(self.caption_state),
            caption_mask=take(self.caption_mask),
            frames=[self.frames[i] for i in index.tolist()],
            latent_mask=take(self.latent_mask),
            log_frames=take(self.log_frames),
        )

    def concat(self, other: EncodedBatch) -> EncodedBatch:
        def cat(x: torch.Tensor | None, y: torch.Tensor | None) -> torch.Tensor | None:
            return None if x is None or y is None else torch.cat([x, y])

        return EncodedBatch(
            text_state=cat(self.text_state, other.text_state),
            text_mask=cat(self.text_mask, other.text_mask),
            speaker_state=cat(self.speaker_state, other.speaker_state),
            speaker_mask=cat(self.speaker_mask, other.speaker_mask),
            caption_state=cat(self.caption_state, other.caption_state),
            caption_mask=cat(self.caption_mask, other.caption_mask),
            frames=self.frames + other.frames,
            latent_mask=cat(self.latent_mask, other.latent_mask),
            log_frames=cat(self.log_frames, other.log_frames),
        )


class ConditionSampler:
    """Samples condition triples from text, reference-voice and caption pools."""

    def __init__(
        self,
        texts: list[str],
        refs: list[tuple[str, torch.Tensor]],  # (speaker id, latent)
        captions: list[str],
        *,
        p_no_ref: float = 0.15,
        p_caption: float = 0.35,
        p_multi_ref: float = 0.25,
        max_ref_frames: int = 750,
        seed: int = 0,
    ) -> None:
        self.texts = texts
        self.refs = refs
        self.captions = captions
        self.p_no_ref = p_no_ref
        self.p_caption = p_caption
        self.p_multi_ref = p_multi_ref
        self.max_ref_frames = max_ref_frames
        self.by_speaker: dict[str, list[int]] = {}
        for index, (speaker, _) in enumerate(refs):
            self.by_speaker.setdefault(speaker, []).append(index)
        self.rng = random.Random(seed)

    def _reference(self) -> torch.Tensor:
        speaker, latent = self.refs[self.rng.randrange(len(self.refs))]
        pieces = [latent]
        if self.rng.random() < self.p_multi_ref:
            same = self.by_speaker[speaker]
            for _ in range(self.rng.randint(1, 4)):
                pieces.append(self.refs[self.rng.choice(same)][1])
        return torch.cat(pieces, dim=0)[: self.max_ref_frames]

    def sample(self, n: int) -> list[ConditionItem]:
        items = []
        for _ in range(n):
            ref = None if self.rng.random() < self.p_no_ref else self._reference()
            caption = self.rng.choice(self.captions) if self.rng.random() < self.p_caption else None
            items.append(ConditionItem(text=self.rng.choice(self.texts), ref=ref, caption=caption))
        return items


def load_texts(paths: list[Path], max_chars: int = 100) -> list[str]:
    texts: list[str] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = normalize_text(line).strip()
            if line and len(line) <= max_chars:
                texts.append(line)
    return texts


def load_refs(paths: list[Path], exclude_corpora: set[str] | None = None) -> list[tuple[str, torch.Tensor]]:
    """Reference latents from ``build_ref_pool.py`` / ``build_invented_voices.py`` files."""
    refs: list[tuple[str, torch.Tensor]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for latent, meta in zip(payload["latents"], payload["meta"], strict=True):
            if exclude_corpora and meta["corpus"] in exclude_corpora:
                continue
            refs.append((str(meta["speaker"]), latent.float()))
    return refs


def _encode_texts(
    tokenizer: PretrainedTextTokenizer, texts: list[str], max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [len(ids) for ids in tokenizer.tokenizer(texts, add_special_tokens=False)["input_ids"]]
    return tokenizer.batch_encode(texts, max_length=max(2, min(max_len, max(lengths) + 1)))


def encode_batch(
    model: TextToLatentRFDiT,
    tokenizer: PretrainedTextTokenizer,
    items: list[ConditionItem],
    *,
    device: torch.device,
    min_seconds: float = 0.5,
    max_seconds: float = 30.0,
    frames: list[int] | None = None,
    grad: bool = False,
) -> EncodedBatch:
    """Encode conditions and predict lengths with ``model``'s encoders and duration predictor.

    ``frames`` fixes the latent lengths (e.g. the teacher's, for a student encoding the
    same items); ``grad`` keeps the autograd graph (training the encoders).
    """
    with torch.set_grad_enabled(grad):
        return _encode_batch(model, tokenizer, items, device, min_seconds, max_seconds, frames)


def _encode_batch(
    model: TextToLatentRFDiT,
    tokenizer: PretrainedTextTokenizer,
    items: list[ConditionItem],
    device: torch.device,
    min_seconds: float,
    max_seconds: float,
    frames: list[int] | None,
) -> EncodedBatch:
    cfg = model.cfg
    batch = len(items)
    texts = [item.text for item in items]
    text_ids, text_mask = _encode_texts(tokenizer, texts, MAX_TEXT_LEN)

    caption_texts = [item.caption or "" for item in items]
    caption_ids, caption_mask = _encode_texts(tokenizer, caption_texts, MAX_CAPTION_LEN)
    has_caption = torch.tensor([item.caption is not None for item in items])
    caption_mask &= has_caption[:, None]

    patch = int(cfg.speaker_patch_size)
    ref_len = max([item.ref.shape[0] for item in items if item.ref is not None] + [patch])
    ref_len = math.ceil(ref_len / patch) * patch
    ref_latent = torch.zeros(batch, ref_len, cfg.patched_latent_dim)
    ref_mask = torch.zeros(batch, ref_len, dtype=torch.bool)
    for i, item in enumerate(items):
        if item.ref is not None:
            ref_latent[i, : item.ref.shape[0]] = item.ref
            ref_mask[i, : item.ref.shape[0]] = True
    has_speaker = ref_mask.any(dim=1)

    dtype = next(model.parameters()).dtype
    text_state, text_mask, speaker_state, speaker_mask, caption_state, caption_mask = model.encode_conditions(
        text_input_ids=text_ids.to(device),
        text_mask=text_mask.to(device),
        ref_latent=ref_latent.to(device, dtype),
        ref_mask=ref_mask.to(device),
        caption_input_ids=caption_ids.to(device),
        caption_mask=caption_mask.to(device),
    )
    features = build_duration_features(
        texts, token_counts=text_mask.sum(dim=1), max_text_len=MAX_TEXT_LEN, has_speaker=has_speaker
    ).to(device)
    log_frames = model.predict_duration_log_frames(
        text_state=text_state,
        text_mask=text_mask,
        speaker_state=speaker_state,
        speaker_mask=speaker_mask,
        caption_state=caption_state,
        caption_mask=caption_mask,
        duration_features=features,
        has_speaker=has_speaker.to(device),
        has_caption=has_caption.to(device),
        detach_condition=not torch.is_grad_enabled(),
    )
    if frames is None:
        min_frames = max(1, math.ceil(min_seconds * FRAMES_PER_SECOND))
        max_frames = max(1, math.floor(max_seconds * FRAMES_PER_SECOND))
        predicted = torch.expm1(log_frames.detach()).float().tolist()
        frames = [max(min_frames, min(max_frames, round(f))) for f in predicted]
    latent_mask = torch.zeros(batch, max(frames), dtype=torch.bool, device=device)
    for i, length in enumerate(frames):
        latent_mask[i, :length] = True
    return EncodedBatch(
        text_state=text_state,
        text_mask=text_mask,
        speaker_state=speaker_state,
        speaker_mask=speaker_mask,
        caption_state=caption_state,
        caption_mask=caption_mask,
        frames=frames,
        latent_mask=latent_mask,
        log_frames=log_frames,
    )


def initial_noise(
    encoded: EncodedBatch, latent_dim: int, generator: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    """Gaussian noise on valid frames, zeros on padding (as ``batch_synth`` pads)."""
    mask = encoded.latent_mask
    noise = torch.randn(
        (mask.shape[0], mask.shape[1], latent_dim), device=mask.device, generator=generator, dtype=torch.float32
    )
    return (noise * mask.unsqueeze(-1)).to(dtype)


@dataclass
class Rollout:
    states: list[torch.Tensor]  # x_t before each step
    times: list[float]
    deltas: list[float]
    outputs: list[torch.Tensor]  # model output u at each state
    final: torch.Tensor  # x_0


@torch.no_grad()
def meanflow_rollout(
    model: TextToLatentRFDiT, encoded: EncodedBatch, x_1: torch.Tensor, num_steps: int = 4
) -> Rollout:
    """Linear MeanFlow sampling from t=1 to 0 (``irodori_tts.meanflow.sample_euler_meanflow``)."""
    batch = x_1.shape[0]
    conditions = encoded.conditions()
    kv_cache = model.build_context_kv_cache(
        text_state=conditions["text_state"],
        speaker_state=conditions["speaker_state"],
        caption_state=conditions["caption_state"],
    )
    schedule = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32).tolist()
    x_t = x_1
    states, times, deltas, outputs = [], [], [], []
    for step in range(num_steps):
        t, t_next = schedule[step], schedule[step + 1]
        u = model.forward_with_encoded_conditions(
            x_t=x_t,
            t=torch.full((batch,), t, device=x_t.device),
            delta_t=torch.full((batch,), t - t_next, device=x_t.device),
            latent_mask=encoded.latent_mask,
            context_kv_cache=kv_cache,
            **conditions,
        )
        states.append(x_t)
        times.append(t)
        deltas.append(t - t_next)
        outputs.append(u)
        x_t = x_t + u * (t_next - t)
    return Rollout(states=states, times=times, deltas=deltas, outputs=outputs, final=x_t)


def masked_utterance_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid frames and channels per utterance, then over the batch."""
    weight = mask.unsqueeze(-1).float()
    per_item = (((pred.float() - target.float()) ** 2) * weight).sum(dim=(1, 2))
    return (per_item / (weight.sum(dim=(1, 2)) * pred.shape[-1]).clamp_min(1.0)).mean()


def masked_relative_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.unsqueeze(-1).float()
    num = (((pred.float() - target.float()) ** 2) * weight).sum()
    den = ((target.float() ** 2) * weight).sum()
    return float((num / den).sqrt())
