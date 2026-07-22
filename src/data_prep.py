"""Download, clean, and split the UMC005 English-Urdu parallel corpus.

Combines the freely-redistributable Quran and Bible sections of UMC005 (the
Penn/Emille sections are excluded by the corpus authors for licensing
reasons) into a single train/dev/test split under data/processed/.
"""
import io
import re
import random
import zipfile
from pathlib import Path
from urllib.request import urlopen

CORPUS_URL = "https://ufal.mff.cuni.cz/umc/005-en-ur/download.php?f=umc005-corpus.zip"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SECTIONS = ["quran", "bible"]
SPLITS = ["train", "dev", "test"]

BOM = "﻿"
WHITESPACE_RE = re.compile(r"\s+")


def download_corpus() -> None:
    """Fetch and extract the UMC005 zip if not already present locally."""
    zip_path = RAW_DIR / "umc005-corpus.zip"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        with urlopen(CORPUS_URL) as resp:
            zip_path.write_bytes(resp.read())
    if not (RAW_DIR / "quran").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW_DIR)


def clean_line(line: str) -> str:
    line = line.replace(BOM, "").strip()
    line = WHITESPACE_RE.sub(" ", line)
    return line


def load_pairs(section: str, split: str) -> list[tuple[str, str]]:
    en_path = RAW_DIR / section / f"{split}.en"
    ur_path = RAW_DIR / section / f"{split}.ur"
    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    ur_lines = ur_path.read_text(encoding="utf-8").splitlines()
    assert len(en_lines) == len(ur_lines), f"{section}/{split}: misaligned line counts"
    return list(zip(en_lines, ur_lines))


def clean_and_filter(pairs: list[tuple[str, str]], min_len=1, max_len=100) -> list[tuple[str, str]]:
    seen = set()
    out = []
    for en, ur in pairs:
        en, ur = clean_line(en), clean_line(ur)
        if not en or not ur:
            continue
        en_wc, ur_wc = len(en.split()), len(ur.split())
        if not (min_len <= en_wc <= max_len and min_len <= ur_wc <= max_len):
            continue
        key = (en, ur)
        if key in seen:
            continue
        seen.add(key)
        out.append((en, ur))
    return out


def write_split(pairs: list[tuple[str, str]], split: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / f"{split}.en").write_text(
        "\n".join(en for en, _ in pairs) + "\n", encoding="utf-8"
    )
    (PROCESSED_DIR / f"{split}.ur").write_text(
        "\n".join(ur for _, ur in pairs) + "\n", encoding="utf-8"
    )


def main(seed: int = 42) -> None:
    download_corpus()
    random.seed(seed)

    for split in SPLITS:
        pairs: list[tuple[str, str]] = []
        for section in SECTIONS:
            pairs.extend(load_pairs(section, split))
        pairs = clean_and_filter(pairs)
        random.shuffle(pairs)
        write_split(pairs, split)
        print(f"{split}: {len(pairs)} sentence pairs")


if __name__ == "__main__":
    main()
