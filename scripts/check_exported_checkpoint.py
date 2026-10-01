#!/usr/bin/env python3
"""Check an exported checkpoint (``export_student_checkpoint.py``) against its sources.

    PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python \
        scripts/check_exported_checkpoint.py ../Irodori-TTS-v4.1-Small-Yomi/model.safetensors \
        --student outputs/yomi_prod/s10_cont/part2/student

1. Text path: the exported checkpoint must give the same text states as the base
   model with the student installed (what all evaluations used).
2. Drift from the base model, as relative L2 per token and cosine similarity:
   captions (the caption path shares the updated backbone), plain sentences,
   and emoji tokens inside sentences (emoji style control goes through the text
   path). For scale, the same metric between the pooled states of different
   captions is also reported.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import random
from pathlib import Path

import torch
from batch_synth import DEFAULT_HF_CHECKPOINT, load_runtime
from irodori_tts.inference_runtime import InferenceRuntime
from student_text import install_student

CAPTIONS = [
    "落ち着いた、近い距離感の女性話者",
    "余裕のある大人の男性。親しい相手に対して、くだけた雰囲気で呆れながらも楽しそうに話している。",
    "落ち着いた自然な声",
    "明るく元気な若い女性が、友達に楽しそうに話しかけている。",
    "低く渋い声の年配の男性が、ゆっくりと語りかける。",
    "ニュースを読み上げるアナウンサーのような、はっきりとした発声。",
    "眠そうな声で、気だるげにつぶやく少年。",
    "怒りを抑えながら、低い声で静かに話す男性。",
    "泣きそうな声で、震えながら話す少女。",
    "電話越しのような、こもった音質の女性の声。",
    "広いホールで響くような、エコーのかかった声。",
    "ささやくように、耳元で優しく話しかける女性。",
    "早口でまくしたてる、慌てた様子の若い男性。",
    "穏やかで温かみのある、絵本を読み聞かせるような声。",
    "自信に満ちた、力強い演説口調の男性。",
    "かわいらしい高めの声で、甘えるように話す女性。",
    "疲れ切った様子で、ため息まじりに話す中年の男性。",
    "驚いた様子で、声を上げる女の子。",
    "冷たく突き放すような、感情を抑えた女性の声。",
    "楽しげに笑いながら話す、陽気な男性。",
    "緊張して声がうわずっている、若い女性の面接の受け答え。",
    "落ち着いたトーンで商品を紹介する、ナレーターの男性。",
    "少しかすれた声の、おっとりした年配の女性。",
    "屋外で風の音が混じる中、大きな声で呼びかける男性。",
    "照れながら、恥ずかしそうに小声で話す少年。",
    "朗らかで張りのある声の、テレビショッピングの司会者。",
    "静かな図書館で、ひそひそ声で話す二十代の女性。",
    "厳かで重々しい、時代劇の武士のような口調。",
    "子どもに言い聞かせるような、ゆっくりとした母親の声。",
    "機械的で抑揚の少ない、淡々とした話し方。",
]

EMOJIS = ["😭", "🤭", "👂", "😠", "😪", "⏩", "🐢", "📢", "😊", "🥺"]


def states(
    runtime: InferenceRuntime, texts: list[str], path: str
) -> list[torch.Tensor]:
    """Per-text (tokens, dim) condition states of the text or caption path."""
    model = runtime.model
    if path == "text":
        tokenizer, encoder, norm = (
            runtime.tokenizer,
            model.text_encoder,
            model.text_norm,
        )
        max_len = runtime.default_text_max_len
    else:
        tokenizer = runtime.caption_tokenizer
        encoder, norm = model.caption_encoder, model.caption_norm
        max_len = runtime.default_caption_max_len
    device = model.text_norm.weight.device
    out: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            chunk = texts[start : start + 32]
            ids, mask = tokenizer.batch_encode(chunk, max_length=max_len)
            ids, mask = ids.to(device), mask.to(device)
            state = norm(encoder(model.pretrained_text_backbone, ids, mask)).float()
            for row, row_mask in zip(state, mask, strict=True):
                out.append(row[row_mask].cpu())
    return out


def drift(
    new: list[torch.Tensor],
    old: list[torch.Tensor],
    select: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    rel, cos = [], []
    for index, (a, b) in enumerate(zip(new, old, strict=True)):
        if select is not None:
            a, b = a[select[index]], b[select[index]]
        rel.append(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)).mean())
        cos.append(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())
    return {
        "rel_l2": round(float(torch.stack(rel).mean()), 5),
        "cosine": round(float(torch.stack(cos).mean()), 5),
    }


def pooled_spread(old: list[torch.Tensor]) -> float:
    """Mean relative L2 between pooled states of different captions."""
    pooled = [s.mean(dim=0) for s in old]
    values = [
        float((a - b).norm() / b.norm()) for a, b in itertools.permutations(pooled, 2)
    ]
    return round(sum(values) / len(values), 5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Exported model.safetensors")
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument(
        "--sentences", type=Path, default=Path("data/yomi/jsut_eval_rows.jsonl")
    )
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.sentences.open(encoding="utf-8")]
    sentences = [row["text"] for row in rows[: args.limit]]
    rng = random.Random(0)
    emoji_sentences = []
    for text in sentences[:200]:
        emoji = rng.choice(EMOJIS)
        cut = rng.randrange(len(text) + 1)
        emoji_sentences.append(text[:cut] + emoji + text[cut:])

    def load(checkpoint: str) -> InferenceRuntime:
        return load_runtime(checkpoint, precision="fp32", text_precision="fp32")

    base = load(DEFAULT_HF_CHECKPOINT)
    base_text = states(base, sentences, "text")
    base_caption = states(base, CAPTIONS, "caption")
    base_emoji = states(base, emoji_sentences, "text")
    install_student(base.model, args.student)
    student_text = states(base, sentences + emoji_sentences, "text")
    del base
    torch.cuda.empty_cache()

    exported = load(args.checkpoint)
    export_text = states(exported, sentences + emoji_sentences, "text")
    export_caption = states(exported, CAPTIONS, "caption")

    max_diff = max(
        float((a - b).abs().max())
        for a, b in zip(export_text, student_text, strict=True)
    )
    # Emoji tokens: tokens of the emoji sentence that do not match the plain one.
    emoji_select = []
    for plain, text in zip(sentences, emoji_sentences, strict=False):
        a, b = (
            exported.tokenizer.batch_encode(
                [t], max_length=exported.default_text_max_len
            )
            for t in (plain, text)
        )
        plain_ids = a[0][0][a[1][0]].tolist()
        ids = b[0][0][b[1][0]].tolist()
        keep = torch.ones(len(ids), dtype=torch.bool)
        matcher = difflib.SequenceMatcher(a=plain_ids, b=ids, autojunk=False)
        for block in matcher.get_matching_blocks():
            keep[block.b : block.b + block.size] = False
        emoji_select.append(keep)
    n = len(sentences)
    report = {
        "text_path_max_abs_diff_vs_student": max_diff,
        "drift_from_base": {
            "caption": drift(export_caption, base_caption),
            "plain_sentences": drift(export_text[:n], base_text),
            "emoji_tokens": drift(export_text[n:], base_emoji, emoji_select),
            "emoji_sentences_all_tokens": drift(export_text[n:], base_emoji),
        },
        "caption_pooled_spread_between_captions": pooled_spread(base_caption),
        "caption_pooled_drift": round(
            sum(
                float((a.mean(0) - b.mean(0)).norm() / b.mean(0).norm())
                for a, b in zip(export_caption, base_caption, strict=True)
            )
            / len(CAPTIONS),
            5,
        ),
        "emoji_tokens_found": int(sum(int(s.sum()) for s in emoji_select)),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
