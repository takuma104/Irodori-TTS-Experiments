#!/usr/bin/env python3
"""Leave-one-layer-out sensitivity of the shared ModernBERT backbone.

Skips one backbone layer at a time and reports the relative L2 error (over
valid tokens) of the text and caption condition states, i.e. the outputs of
``text_norm(text_encoder(...))`` and ``caption_norm(caption_encoder(...))``,
against the unmodified model on JSUT sentences.
"""

from __future__ import annotations

import argparse
import json

import torch
from common import OUTPUT_DIR, jsut_sentences, load_model
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.text_normalization import normalize_text


@torch.inference_mode()
def condition_states(
    model: TextToLatentRFDiT, ids: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone = model.pretrained_text_backbone
    text_state = model.text_norm(model.text_encoder(backbone, ids, mask))
    caption_state = model.caption_norm(model.caption_encoder(backbone, ids, mask))
    return text_state, caption_state


def relative_error(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.unsqueeze(-1).float()
    num = (((pred - ref) ** 2) * weight).sum()
    den = ((ref**2) * weight).sum()
    return float((num / den).sqrt())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sentences", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "bert_layer_drop.json"))
    args = parser.parse_args()

    model, tokenizer = load_model("RF")
    texts = [normalize_text(text) for text in jsut_sentences(seed=0)[: args.num_sentences]]
    batches = []
    for start in range(0, len(texts), args.batch_size):
        ids, mask = tokenizer.batch_encode(texts[start : start + args.batch_size], max_length=256)
        batches.append((ids.cuda(), mask.cuda()))
    reference = [condition_states(model, ids, mask) for ids, mask in batches]

    results: dict[int, dict[str, float | str]] = {}
    for index, layer in enumerate(model.pretrained_text_backbone.backbone.layers):
        original_forward = layer.forward
        layer.forward = lambda hidden_states, *_, **__: hidden_states
        text_errors, caption_errors = [], []
        for (ids, mask), (ref_text, ref_caption) in zip(batches, reference, strict=True):
            text_state, caption_state = condition_states(model, ids, mask)
            text_errors.append(relative_error(text_state, ref_text, mask))
            caption_errors.append(relative_error(caption_state, ref_caption, mask))
        layer.forward = original_forward
        results[index] = {
            "attention_type": layer.attention_type,
            "text_rel_err": sum(text_errors) / len(text_errors),
            "caption_rel_err": sum(caption_errors) / len(caption_errors),
        }
        print(index, results[index], flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
