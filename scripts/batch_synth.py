"""Batched Irodori-TTS synthesis for evaluation and data generation.

``InferenceRuntime.synthesize`` handles one text per call, which is dominated by
kernel-launch overhead on a single GPU. This module samples many texts at once:
texts are sorted by predicted length, padded, and sampled together with a latent
mask. Each item uses its own seed for the initial noise, generated exactly like
the runtime does for a single request, so results match one-at-a-time synthesis
up to numerical noise.

Only the default RF path is supported (independent CFG, no caption, a single
shared reference or no reference). Run it inside the Irodori-TTS environment
(``PYTHONPATH=Irodori-TTS``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from irodori_tts.attention import _SDPA_PRIORITY
from irodori_tts.codec import unpatchify_latent
from irodori_tts.duration import build_duration_features
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    find_flattening_point,
)
from irodori_tts.text_normalization import normalize_text
from torch.nn.attention import SDPBackend

DEFAULT_HF_CHECKPOINT = "Aratako/Irodori-TTS-v4.1-Small"


def set_sdpa_backend(name: str) -> None:
    """Select the SDPA priority used by Irodori's masked attention.

    ``cudnn`` is Irodori's default. On an RTX 5090 in bf16 it rebuilds its
    execution plan for every new sequence length, which made per-sentence
    synthesis about 2.5x slower than fp32. ``efficient`` avoids that.
    """
    priorities = {
        "cudnn": [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    }
    _SDPA_PRIORITY[:] = priorities[name]


TEXT_MODULES = (
    "pretrained_text_backbone",
    "text_encoder",
    "text_norm",
    "caption_encoder",
    "caption_norm",
)


def load_runtime(
    hf_checkpoint: str = DEFAULT_HF_CHECKPOINT,
    *,
    device: str = "cuda",
    precision: str = "bf16",
    text_precision: str | None = None,
    codec_precision: str | None = None,
) -> InferenceRuntime:
    """Load the runtime; ``precision`` applies to the DiT and the other modules.

    ``text_precision="fp32"`` keeps the text/caption encoder (ModernBERT and the
    projectors) in fp32 while the DiT runs in ``precision``. ``codec_precision``
    defaults to ``precision``.
    """
    # A local model.safetensors (e.g. an exported student) or a Hugging Face repo id.
    checkpoint = (
        hf_checkpoint
        if Path(hf_checkpoint).is_file()
        else download_hf_checkpoint(hf_checkpoint)
    )
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(checkpoint),
            model_device=device,
            model_precision=precision,
            codec_device=device,
            codec_precision=codec_precision or precision,
        )
    )
    if text_precision == "fp32":
        for name in TEXT_MODULES:
            module = getattr(runtime.model, name, None)
            if module is not None:
                module.float()
    return runtime


@dataclass(frozen=True)
class SynthItem:
    key: str
    text: str
    seed: int


@dataclass(frozen=True)
class SynthOutput:
    key: str
    latent: torch.Tensor  # (frames, latent_dim), before tail trimming
    audio: torch.Tensor  # (channels, samples), trimmed and watermarked
    sample_rate: int


@dataclass(frozen=True)
class SamplingSettings:
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    duration_scale: float = 1.0
    min_seconds: float = 0.5
    max_seconds: float = 30.0
    trim_tail: bool = True
    watermark: bool = True


class BatchSynthesizer:
    def __init__(
        self,
        runtime: InferenceRuntime,
        *,
        ref_wav: str | None,
        settings: SamplingSettings | None = None,
    ) -> None:
        if runtime.model_cfg.flow_parameterization != "rf_velocity":
            raise ValueError("BatchSynthesizer supports RF checkpoints only.")
        if not runtime.model_cfg.use_speaker_condition_resolved:
            raise ValueError(
                "BatchSynthesizer expects a speaker-conditioned checkpoint."
            )
        self.runtime = runtime
        self.model = runtime.model
        self.settings = settings if settings is not None else SamplingSettings()
        self.device = runtime.model_device
        # The DiT dtype; the text encoder may run in fp32 (see load_runtime).
        self.dtype = self.model.in_proj.weight.dtype
        self.hop_length = int(runtime.codec.model.hop_length)
        self.sample_rate = int(runtime.codec.sample_rate)
        self.text_max_len = int(runtime.default_text_max_len)
        self.caption_max_len = int(runtime.default_caption_max_len)
        self.no_ref = ref_wav is None
        with torch.inference_mode():
            self.ref_latent, self.ref_mask = runtime._load_reference_latent(
                req=SamplingRequest(text="", ref_wav=ref_wav, no_ref=self.no_ref),
                batch_size=1,
                messages=[],
            )
        self.ref_latent = self.ref_latent.to(self.dtype)
        self.cfg_scale_speaker = (
            0.0 if self.no_ref else float(self.settings.cfg_scale_speaker)
        )

    def _encode(
        self, texts: Sequence[str]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Encode conditions; also returns duration features for the batch."""
        batch = len(texts)
        text_ids, text_mask = self.runtime.tokenizer.batch_encode(
            list(texts), max_length=self.text_max_len
        )
        text_ids = text_ids.to(self.device)
        text_mask = text_mask.to(self.device)
        caption_ids = None
        caption_mask = None
        if self.runtime.model_cfg.use_caption_condition:
            caption_ids, caption_mask = self.runtime.caption_tokenizer.batch_encode(
                [""] * batch, max_length=self.caption_max_len
            )
            caption_mask.zero_()
            caption_ids = caption_ids.to(self.device)
            caption_mask = caption_mask.to(self.device)
        ref_latent = self.ref_latent.expand(batch, -1, -1)
        ref_mask = self.ref_mask.expand(batch, -1)
        (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        ) = self.model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
        )
        text_state = text_state.to(self.dtype)
        if caption_state is not None:
            caption_state = caption_state.to(self.dtype)
        duration_features = build_duration_features(
            list(texts),
            token_counts=text_mask.sum(dim=1),
            max_text_len=self.text_max_len,
            has_speaker=ref_mask.any(dim=1),
        ).to(self.device)
        return (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            duration_features,
        )

    @torch.inference_mode()
    def predict_latent_steps(self, texts: Sequence[str]) -> list[int]:
        """Predicted latent frame counts, clamped like ``InferenceRuntime.synthesize``."""
        normalized = [normalize_text(text).strip() for text in texts]
        (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            duration_features,
        ) = self._encode(normalized)
        batch = len(normalized)
        has_caption = None
        if self.runtime.model_cfg.use_caption_condition:
            has_caption = torch.zeros((batch,), dtype=torch.bool, device=self.device)
        pred_log_frames = self.model.predict_duration_log_frames(
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
            duration_features=duration_features,
            has_speaker=self.ref_mask.expand(batch, -1).any(dim=1),
            has_caption=has_caption,
        )
        frames = torch.expm1(pred_log_frames).float().reshape(batch).tolist()
        settings = self.settings
        min_frames = max(
            1, math.ceil(settings.min_seconds * self.sample_rate / self.hop_length)
        )
        max_frames = max(
            1, math.floor(settings.max_seconds * self.sample_rate / self.hop_length)
        )
        return [
            max(min_frames, min(max_frames, round(f * settings.duration_scale)))
            for f in frames
        ]

    @torch.inference_mode()
    def sample_latents(
        self,
        texts: Sequence[str],
        seeds: Sequence[int],
        latent_steps: Sequence[int],
        *,
        pad_multiple: int = 1,
    ) -> list[torch.Tensor]:
        """Sample latents (frames, latent_dim) for a batch of texts of different lengths."""
        cfg = self.runtime.model_cfg
        normalized = [normalize_text(text).strip() for text in texts]
        batch = len(normalized)
        patch = int(cfg.latent_patch_size)
        patched_steps = [math.ceil(steps / patch) for steps in latent_steps]
        max_len = max(patched_steps)
        max_len = math.ceil(max_len / pad_multiple) * pad_multiple
        latent_dim = int(cfg.patched_latent_dim)

        x_t = torch.zeros(
            (batch, max_len, latent_dim), device=self.device, dtype=self.dtype
        )
        latent_mask = torch.zeros(
            (batch, max_len), device=self.device, dtype=torch.bool
        )
        for i, (seed, length) in enumerate(zip(seeds, patched_steps, strict=True)):
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            x_t[i, :length] = torch.randn(
                (1, length, latent_dim),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )[0]
            latent_mask[i, :length] = True

        (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            _,
        ) = self._encode(normalized)
        branches: list[tuple[str, float]] = []
        if self.settings.cfg_scale_text > 0:
            branches.append(("text", float(self.settings.cfg_scale_text)))
        if self.cfg_scale_speaker > 0:
            branches.append(("speaker", self.cfg_scale_speaker))
        mult = 1 + len(branches)

        def _uncond(
            name: str, which: str
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            state, mask = {
                "text": (text_state, text_mask),
                "speaker": (speaker_state, speaker_mask),
                "caption": (caption_state, caption_mask),
            }[which]
            if name != which or state is None or mask is None:
                return state, mask
            return torch.zeros_like(state), torch.zeros_like(mask)

        cfg_states: dict[str, list[torch.Tensor]] = {}
        for which in ("text", "speaker", "caption"):
            parts = [_uncond("cond", which)] + [
                _uncond(name, which) for name, _ in branches
            ]
            if parts[0][0] is None:
                continue
            cfg_states[which + "_state"] = [p[0] for p in parts]
            cfg_states[which + "_mask"] = [p[1] for p in parts]

        def _cat(name: str) -> torch.Tensor | None:
            values = cfg_states.get(name)
            return None if values is None else torch.cat(values, dim=0)

        cfg_text_state = _cat("text_state")
        cfg_speaker_state = _cat("speaker_state")
        cfg_caption_state = _cat("caption_state")
        kv_cond = self.model.build_context_kv_cache(
            text_state=text_state,
            speaker_state=speaker_state,
            caption_state=caption_state,
        )
        kv_cfg = self.model.build_context_kv_cache(
            text_state=cfg_text_state,
            speaker_state=cfg_speaker_state,
            caption_state=cfg_caption_state,
        )
        cfg_masks = {
            "text_mask": _cat("text_mask"),
            "speaker_mask": _cat("speaker_mask"),
            "caption_mask": _cat("caption_mask"),
        }
        cfg_latent_mask = latent_mask.repeat(mult, 1)

        num_steps = int(self.settings.num_steps)
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)
        t_schedule = (1.0 - u) * 0.999
        for step in range(num_steps):
            t = t_schedule[step]
            t_next = t_schedule[step + 1]
            tt = torch.full((batch,), t, device=self.device, dtype=self.dtype)
            use_cfg = bool(branches) and (
                self.settings.cfg_min_t <= t.item() <= self.settings.cfg_max_t
            )
            if use_cfg:
                v_out = self.model.forward_with_encoded_conditions(
                    x_t=x_t.repeat(mult, 1, 1),
                    t=tt.repeat(mult),
                    text_state=cfg_text_state,
                    text_mask=cfg_masks["text_mask"],
                    speaker_state=cfg_speaker_state,
                    speaker_mask=cfg_masks["speaker_mask"],
                    caption_state=cfg_caption_state,
                    caption_mask=cfg_masks["caption_mask"],
                    latent_mask=cfg_latent_mask,
                    context_kv_cache=kv_cfg,
                )
                chunks = v_out.chunk(mult, dim=0)
                v = chunks[0]
                for (_, scale), chunk in zip(branches, chunks[1:], strict=True):
                    v = v + scale * (chunks[0] - chunk)
            else:
                v = self.model.forward_with_encoded_conditions(
                    x_t=x_t,
                    t=tt,
                    text_state=text_state,
                    text_mask=text_mask,
                    speaker_state=speaker_state,
                    speaker_mask=speaker_mask,
                    caption_state=caption_state,
                    caption_mask=caption_mask,
                    latent_mask=latent_mask,
                    context_kv_cache=kv_cond,
                )
            x_t = x_t + v * (t_next - t)

        latents: list[torch.Tensor] = []
        for i, steps in enumerate(latent_steps):
            z = unpatchify_latent(
                x_t[i : i + 1, : patched_steps[i]],
                patch_size=patch,
                latent_dim=cfg.latent_dim,
            )
            latents.append(z[0, :steps])
        return latents

    @torch.inference_mode()
    def decode(self, latents: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Decode, trim the flat tail, and watermark like ``InferenceRuntime.synthesize``."""
        audios: list[torch.Tensor] = []
        for z in latents:
            audio = self.runtime.codec.decode_latent(z[None]).cpu()[0]
            max_samples = int(z.shape[0]) * self.hop_length
            if self.settings.trim_tail:
                flattening = find_flattening_point(
                    z, window_size=20, std_threshold=0.05, mean_threshold=0.1
                )
                if flattening > 0:
                    max_samples = min(max_samples, int(flattening) * self.hop_length)
            audios.append(audio[:, :max_samples])
        if self.settings.watermark and self.runtime.watermarker.ready:
            audios = self.runtime.watermarker.encode_batch(
                audios, sample_rate=self.sample_rate
            )
        return audios

    def synthesize(
        self,
        items: Sequence[SynthItem],
        *,
        max_batch_size: int = 32,
        max_batch_frames: int = 8192,
        pad_multiple: int = 1,
        duration_batch_size: int = 256,
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[SynthOutput]:
        """Yield outputs batch by batch (sorted by length, not in input order)."""
        steps: list[int] = []
        for start in range(0, len(items), duration_batch_size):
            chunk = items[start : start + duration_batch_size]
            steps.extend(self.predict_latent_steps([item.text for item in chunk]))
        order = sorted(range(len(items)), key=lambda i: steps[i])
        done = 0
        for batch in _length_batches(order, steps, max_batch_size, max_batch_frames):
            batch_items = [items[i] for i in batch]
            latents = self.sample_latents(
                [item.text for item in batch_items],
                [item.seed for item in batch_items],
                [steps[i] for i in batch],
                pad_multiple=pad_multiple,
            )
            audios = self.decode(latents)
            for item, latent, audio in zip(batch_items, latents, audios, strict=True):
                yield SynthOutput(
                    key=item.key,
                    latent=latent.cpu(),
                    audio=audio,
                    sample_rate=self.sample_rate,
                )
            done += len(batch)
            if progress is not None:
                progress(done, len(items))


def _length_batches(
    order: Sequence[int],
    steps: Sequence[int],
    max_batch_size: int,
    max_batch_frames: int,
) -> Iterator[list[int]]:
    """Group length-sorted indices so that batch_size * max_length stays within budget."""
    batch: list[int] = []
    for index in order:
        candidate = batch + [index]
        if batch and (
            len(candidate) > max_batch_size
            or len(candidate) * max(steps[i] for i in candidate) > max_batch_frames
        ):
            yield batch
            candidate = [index]
        batch = candidate
    if batch:
        yield batch
