#!/usr/bin/env python3
"""Build the reading lexicon for the yomi fine-tuning data (plan §4.1-4.3, §4.6).

    uv run python scripts/build_yomi_lexicon.py --output-dir data/yomi

Inputs:

- JMdict (``data/jmdict/JMdict_e.gz``) for words, readings, frequency tags,
  and sense tags (e.g. ``Buddh``).
- JmdictFurigana (``data/jmdict/JmdictFurigana.json``) for the reading of each
  kanji inside a word (従容 -> 従=しょう, 容=よう).
- The Joyo reading table, taken from the JKYB-Parakeet keys and their accepted
  readings (the 4,512 (kanji, reading) pairs). Only the table is used here; JKYB
  sentences are used only to find the JKYB target words to exclude.

Outputs (``--output-dir``):

- ``lexicon.jsonl``: one row per (word, reading) with each kanji's reading,
  its class (on / kun / jukujikun / nonjoyo), tokenization, frequency tier,
  homograph flag, bucket labels, and the JKYB split.
- ``kanji_stats.json``: frequency-weighted reading distribution per kanji.
- ``summary.md``: counts per bucket.

JKYB split (plan §4.6): each Joyo (kanji, reading) pair is assigned to group A or
B by a hash. Words that appear as a JKYB target word of a group-B pair are marked
``jkyb_excluded`` and must not be used for training.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from analyze_jkyb_errors import strip_sound_change
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DATASET_REPO = "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet"
DATASET_FILENAME = "data/common_kanji_source.jsonl"
TOKENIZER_REPO = "Aratako/Irodori-TTS-v4.1-Small"
TOKENIZER_FILENAME = "tokenizer/tokenizer.json"
KANJI = re.compile(r"[㐀-鿿々]")
WORD_CHARS = re.compile(r"[㐀-鿿々〆ヶぁ-ゖァ-ヺー]+")
ENTITY = re.compile(r"&([\w.-]+);")
XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}
# Kanji-form tags that mark irregular, rare, outdated, or search-only spellings.
SKIP_KANJI_INFO = {"iK", "rK", "oK", "sK", "io"}
TIER_WEIGHTS = {1: 10.0, 2: 3.0, 3: 1.0}

BUCKET_LABELS = {
    "A": "分割される熟語の音読み（単独だと訓が優勢な漢字）",
    "B": "音の種類（最頻でない方の音）",
    "C": "訓読み（音が優勢な漢字）",
    "D": "訓の種類（最頻でない方の訓）",
    "E": "同形異音語",
    "F": "熟字訓・付表の語",
}


def to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --- Joyo table and JKYB split --------------------------------------------------


@dataclass
class JoyoReading:
    key: str  # e.g. "固_かためる"
    kanji: str
    category: str  # on_yomi / kun_yomi / joyo_appendix_reading
    stems: set[str] = field(default_factory=set)  # katakana, target span only


def joyo_key(row: dict[str, Any]) -> str:
    return row["key"].rsplit("_", 1)[0]


def jkyb_group(key: str) -> str:
    return "A" if hashlib.md5(key.encode()).digest()[0] % 2 == 0 else "B"


def target_kanji(row: dict[str, Any]) -> str:
    return re.search(r"<([^>]+)>", row["tagged_text"]).group(1)


def jkyb_target_word(row: dict[str, Any]) -> str:
    """Kanji run containing the target, plus one okurigana for kun readings."""
    tagged = row["tagged_text"]
    start = tagged.index("<")
    end = tagged.index(">") - 1
    text = tagged.replace("<", "").replace(">", "")
    while start > 0 and KANJI.match(text[start - 1]):
        start -= 1
    while end < len(text) and KANJI.match(text[end]):
        end += 1
    if (
        row["reading_category"] == "kun_yomi"
        and end < len(text)
        and "ぁ" <= text[end] <= "ゟ"
    ):
        end += 1
    return text[start:end]


def load_joyo(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[JoyoReading]], dict[str, str]]:
    """Per-kanji Joyo readings, and appendix words (付表の語) -> reading."""
    readings: dict[str, JoyoReading] = {}
    appendix: dict[str, str] = {}
    for row in rows:
        key = joyo_key(row)
        kanji = target_kanji(row)
        if row["reading_category"] == "joyo_appendix_reading":
            appendix[kanji] = row["readings"]["natural"][0]
            continue
        entry = readings.setdefault(
            key, JoyoReading(key=key, kanji=kanji, category=row["reading_category"])
        )
        entry.stems.update(row["readings"]["natural"])
    by_kanji: dict[str, list[JoyoReading]] = defaultdict(list)
    for entry in readings.values():
        by_kanji[entry.kanji].append(entry)
    return by_kanji, appendix


def jkyb_words(rows: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    """Group (A/B) -> target kanji -> JKYB target words containing it."""
    words: dict[str, dict[str, set[str]]] = {
        "A": defaultdict(set),
        "B": defaultdict(set),
    }
    for row in rows:
        target = target_kanji(row)
        if row["reading_category"] == "joyo_appendix_reading":
            # The whole appendix word (e.g. 海女) is the target.
            words[jkyb_group(joyo_key(row))][target].add(target)
            continue
        words[jkyb_group(joyo_key(row))][target].add(jkyb_target_word(row))
    return words


def jkyb_word_readings(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, set[tuple[str, str]]]]:
    """Group (A/B) -> target kanji -> (JKYB target word, accepted reading) pairs."""
    pairs: dict[str, dict[str, set[tuple[str, str]]]] = {
        "A": defaultdict(set),
        "B": defaultdict(set),
    }
    for row in rows:
        target = target_kanji(row)
        word = (
            target
            if row["reading_category"] == "joyo_appendix_reading"
            else jkyb_target_word(row)
        )
        for reading in row["readings"]["natural"] + row["readings"]["marginal"]:
            pairs[jkyb_group(joyo_key(row))][target].add((word, reading))
    return pairs


# --- JMdict ---------------------------------------------------------------------


@dataclass
class Entry:
    seq: str
    kanji_forms: list[tuple[str, list[str], list[str]]]  # (keb, ke_inf, ke_pri)
    readings: list[
        tuple[str, list[str], list[str], bool]
    ]  # (reb, re_restr, re_pri, nokanji)
    pos: list[str]
    misc: list[str]
    fields: list[str]
    gloss: list[str]


def parse_jmdict(path: Path) -> list[Entry]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    body = text[text.index("<JMdict>") :]
    # Keep entity names (e.g. &Buddh;) as plain codes instead of expanding them.
    body = ENTITY.sub(
        lambda m: m.group(0) if m.group(1) in XML_ENTITIES else m.group(1), body
    )
    root = ET.fromstring(body)
    entries: list[Entry] = []
    for node in root.iter("entry"):
        kanji_forms = [
            (
                k.findtext("keb", ""),
                [x.text or "" for x in k.findall("ke_inf")],
                [x.text or "" for x in k.findall("ke_pri")],
            )
            for k in node.findall("k_ele")
        ]
        if not kanji_forms:
            continue
        readings = [
            (
                r.findtext("reb", ""),
                [x.text or "" for x in r.findall("re_restr")],
                [x.text or "" for x in r.findall("re_pri")],
                r.find("re_nokanji") is not None,
            )
            for r in node.findall("r_ele")
        ]
        senses = node.findall("sense")
        entries.append(
            Entry(
                seq=node.findtext("ent_seq", ""),
                kanji_forms=kanji_forms,
                readings=readings,
                pos=sorted({x.text or "" for s in senses for x in s.findall("pos")}),
                misc=sorted({x.text or "" for s in senses for x in s.findall("misc")}),
                fields=sorted(
                    {x.text or "" for s in senses for x in s.findall("field")}
                ),
                gloss=[x.text or "" for s in senses[:2] for x in s.findall("gloss")][
                    :4
                ],
            )
        )
    return entries


def frequency_tier(priorities: list[str]) -> int:
    """1: top ~6k (nf01-12 or ichi1/news1/spec1), 2: other tagged, 3: untagged."""
    nf = [int(p[2:]) for p in priorities if p.startswith("nf")]
    if nf and min(nf) <= 12:
        return 1
    if not nf and any(p in {"ichi1", "news1", "spec1"} for p in priorities):
        return 1
    return 2 if priorities else 3


# --- Per-kanji reading classification -------------------------------------------


@dataclass
class KanjiReading:
    index: int  # character index in the word
    kanji: str  # one kanji, or a multi-kanji group for jukujikun
    reading: str  # katakana
    kind: str  # on / kun / jukujikun / appendix / nonjoyo
    joyo_key: str | None
    exact: bool  # matched without allowing rendaku / sokuon changes


def classify(
    kanji: str, reading: str, joyo: dict[str, list[JoyoReading]]
) -> tuple[str, str | None, bool]:
    candidates = joyo.get(kanji, [])
    for exact in (True, False):
        for entry in candidates:
            stems = entry.stems
            if exact:
                matched = reading in stems
            else:
                matched = strip_sound_change(reading) in {
                    strip_sound_change(s) for s in stems
                }
            if matched:
                kind = "on" if entry.category == "on_yomi" else "kun"
                return kind, entry.key, exact
    return "nonjoyo", None, False


def split_furigana(
    furigana: list[dict[str, str]], joyo: dict[str, list[JoyoReading]]
) -> list[KanjiReading]:
    out: list[KanjiReading] = []
    index = 0
    for part in furigana:
        ruby = part["ruby"]
        rt = part.get("rt")
        if rt and any(KANJI.match(ch) for ch in ruby):
            reading = to_katakana(rt)
            if len(ruby) == 1:
                kind, key, exact = classify(ruby, reading, joyo)
                out.append(KanjiReading(index, ruby, reading, kind, key, exact))
            else:
                out.append(KanjiReading(index, ruby, reading, "jukujikun", None, False))
        index += len(ruby)
    return out


# --- Lexicon rows -----------------------------------------------------------------


@dataclass
class LexiconRow:
    word: str
    reading: str  # hiragana
    seq: str
    tier: int
    pos: list[str]
    misc: list[str]
    fields: list[str]
    gloss: list[str]
    kanji_readings: list[KanjiReading]
    tokens: list[str]
    single_char_kanji: list[int]  # indices of kanji tokenized as one-character tokens
    homograph_readings: list[str] = field(default_factory=list)
    buckets: list[str] = field(default_factory=list)
    bucket_targets: dict[str, list[int]] = field(default_factory=dict)
    jkyb_groups: list[str] = field(default_factory=list)
    jkyb_word: bool = False
    jkyb_excluded: bool = False
    # Finer exclusion: only when the word also takes a group-B JKYB reading
    # (外 ほか stays usable although 外 そと is a group-B target).
    jkyb_excluded_reading: bool = False


def build_rows(
    entries: list[Entry],
    furigana: dict[tuple[str, str], list[dict[str, str]]],
    joyo: dict[str, list[JoyoReading]],
    appendix: dict[str, str],
    tokenizer: Tokenizer,
) -> list[LexiconRow]:
    rows: list[LexiconRow] = []
    for entry in entries:
        for keb, ke_inf, ke_pri in entry.kanji_forms:
            if (
                SKIP_KANJI_INFO & set(ke_inf)
                or not KANJI.search(keb)
                or not WORD_CHARS.fullmatch(keb)
            ):
                continue
            for reb, restr, re_pri, nokanji in entry.readings:
                if nokanji or (restr and keb not in restr):
                    continue
                parts = furigana.get((keb, reb))
                if parts is None:
                    continue
                kanji_readings = split_furigana(parts, joyo)
                if keb in appendix and to_katakana(reb) == appendix[keb]:
                    kanji_readings = [
                        KanjiReading(0, keb, appendix[keb], "appendix", None, True)
                    ]
                encoding = tokenizer.encode(keb, add_special_tokens=False)
                single = [
                    i
                    for i, ch in enumerate(keb)
                    if KANJI.match(ch)
                    and any(s <= i < e and e - s == 1 for s, e in encoding.offsets)
                ]
                rows.append(
                    LexiconRow(
                        word=keb,
                        reading=reb,
                        seq=entry.seq,
                        tier=frequency_tier(ke_pri + re_pri),
                        pos=entry.pos,
                        misc=entry.misc,
                        fields=entry.fields,
                        gloss=entry.gloss,
                        kanji_readings=kanji_readings,
                        tokens=encoding.tokens,
                        single_char_kanji=single,
                    )
                )
    return rows


def kanji_statistics(rows: list[LexiconRow]) -> dict[str, dict[str, Any]]:
    """Frequency-weighted distribution of readings and on/kun per kanji."""
    by_reading: dict[str, Counter[str]] = defaultdict(Counter)
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        weight = TIER_WEIGHTS[row.tier]
        for kr in row.kanji_readings:
            if kr.kind in ("on", "kun") and kr.joyo_key is not None:
                by_reading[kr.kanji][kr.joyo_key] += weight
                by_kind[kr.kanji][kr.kind] += weight
    stats: dict[str, dict[str, Any]] = {}
    for kanji, readings in by_reading.items():
        kinds = by_kind[kanji]
        stats[kanji] = {
            "readings": dict(readings.most_common()),
            "kinds": dict(kinds),
            "default_key": readings.most_common(1)[0][0],
            "default_kind": kinds.most_common(1)[0][0],
            "on_share": kinds["on"] / max(sum(kinds.values()), 1e-9),
        }
    return stats


def assign_buckets(
    rows: list[LexiconRow],
    stats: dict[str, dict[str, Any]],
    joyo: dict[str, list[JoyoReading]],
) -> None:
    readings_by_word: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.tier < 3 or "arch" not in row.misc:
            readings_by_word[row.word].add(row.reading)
    kind_of_key = {e.key: e.category for entries in joyo.values() for e in entries}
    for row in rows:
        others = sorted(readings_by_word[row.word] - {row.reading})
        row.homograph_readings = others
        targets: dict[str, list[int]] = defaultdict(list)
        if others:
            targets["E"] = [kr.index for kr in row.kanji_readings]
        for kr in row.kanji_readings:
            if kr.kind in ("jukujikun", "appendix"):
                targets["F"].append(kr.index)
                continue
            if kr.kind not in ("on", "kun") or kr.kanji not in stats:
                continue
            s = stats[kr.kanji]
            same_kind = {
                k: w
                for k, w in s["readings"].items()
                if kind_of_key.get(k) == ("on_yomi" if kr.kind == "on" else "kun_yomi")
            }
            top_same_kind = max(same_kind, key=same_kind.get) if same_kind else None
            split = kr.index in row.single_char_kanji and len(row.word) > 1
            if kr.kind == "on" and s["on_share"] < 0.5 and split:
                targets["A"].append(kr.index)
            if kr.kind == "on" and len(same_kind) > 1 and kr.joyo_key != top_same_kind:
                targets["B"].append(kr.index)
            if kr.kind == "kun" and s["on_share"] >= 0.5:
                targets["C"].append(kr.index)
            if kr.kind == "kun" and len(same_kind) > 1 and kr.joyo_key != top_same_kind:
                targets["D"].append(kr.index)
        row.bucket_targets = {b: sorted(set(v)) for b, v in sorted(targets.items())}
        row.buckets = sorted(row.bucket_targets)


def matches_jkyb_word(word: str, jkyb_word: str) -> bool:
    """Whether a lexicon word is (part of) a JKYB target word.

    Single-kanji words match only when they are the whole kanji run of the JKYB
    word (氏 in 氏が), not any kanji inside a longer run (上 in 今上陛下).
    """
    if len(word) >= 2:
        return word in jkyb_word
    return word == re.sub(r"[ぁ-ゟ]+$", "", jkyb_word)


def mark_jkyb(
    rows: list[LexiconRow],
    words: dict[str, dict[str, set[str]]],
    word_readings: dict[str, dict[str, set[tuple[str, str]]]],
) -> None:
    for row in rows:
        row.jkyb_excluded_reading = any(
            matches_jkyb_word(row.word, jkyb_word)
            and strip_sound_change(kr.reading) == strip_sound_change(reading)
            for kr in row.kanji_readings
            for jkyb_word, reading in word_readings["B"].get(kr.kanji, ())
        )
        groups: set[str] = set()
        for group in ("A", "B"):
            for kr in row.kanji_readings:
                if any(
                    matches_jkyb_word(row.word, jkyb_word)
                    for jkyb_word in words[group].get(kr.kanji, ())
                ):
                    groups.add(group)
        row.jkyb_groups = sorted(groups)
        row.jkyb_word = bool(groups)
        row.jkyb_excluded = "B" in groups


def row_to_json(row: LexiconRow) -> dict[str, Any]:
    data = asdict(row)
    data["kanji_readings"] = [asdict(kr) for kr in row.kanji_readings]
    return data


def summarize(rows: list[LexiconRow]) -> str:
    lines = ["# 読み語彙表の集計", ""]
    lines.append(f"- 語（表記, 読み）の数: {len(rows):,}")
    lines.append(
        f"- JKYB の対象語に一致: {sum(r.jkyb_word for r in rows):,}"
        f"（うち B 群で学習から除外: {sum(r.jkyb_excluded for r in rows):,}）"
    )
    kinds = Counter(kr.kind for r in rows for kr in r.kanji_readings)
    lines.append(
        "- 漢字ごとの読みの分類: "
        + ", ".join(f"{k} {v:,}" for k, v in kinds.most_common())
    )
    lines += [
        "",
        "| バケット | 語数 | 頻度tier1 | tier2 | tier3 | 分割あり | 除外後 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket, label in BUCKET_LABELS.items():
        in_bucket = [r for r in rows if bucket in r.buckets]
        tiers = Counter(r.tier for r in in_bucket)
        split = sum(
            any(i in r.single_char_kanji for i in r.bucket_targets[bucket])
            for r in in_bucket
        )
        kept = sum(not r.jkyb_excluded for r in in_bucket)
        lines.append(
            f"| {bucket}. {label} | {len(in_bucket):,} | {tiers[1]:,} | {tiers[2]:,} | "
            f"{tiers[3]:,} | {split:,} | {kept:,} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jmdict", type=Path, default=Path("data/jmdict/JMdict_e.gz"))
    parser.add_argument(
        "--furigana", type=Path, default=Path("data/jmdict/JmdictFurigana.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/yomi"))
    args = parser.parse_args()

    jkyb_rows = read_jsonl(
        Path(hf_hub_download(DATASET_REPO, DATASET_FILENAME, repo_type="dataset"))
    )
    joyo, appendix = load_joyo(jkyb_rows)
    furigana = {
        (item["text"], item["reading"]): item["furigana"]
        for item in json.loads(args.furigana.read_text(encoding="utf-8-sig"))
    }
    tokenizer = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, TOKENIZER_FILENAME))
    entries = parse_jmdict(args.jmdict)
    rows = build_rows(entries, furigana, joyo, appendix, tokenizer)
    stats = kanji_statistics(rows)
    assign_buckets(rows, stats, joyo)
    mark_jkyb(rows, jkyb_words(jkyb_rows), jkyb_word_readings(jkyb_rows))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "lexicon.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row_to_json(row), ensure_ascii=False) + "\n")
    (args.output_dir / "kanji_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    summary = summarize(rows)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
