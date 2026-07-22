"""Train independent Byte-Level BPE tokenizers for English and Urdu.

Each language gets its own tokenizer (different scripts/morphology), trained
on the training split only, and saved as a single-file HF `tokenizers` JSON
that `dataset.py` and the model scripts load at run time.
"""
from tokenizers import ByteLevelBPETokenizer

from utils import PROCESSED_DIR, SPECIAL_TOKENS, TOKENIZER_DIR, VOCAB_SIZE


def train_tokenizer(lang: str) -> None:
    train_file = str(PROCESSED_DIR / f"train.{lang}")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[train_file],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_DIR / f"{lang}.json"))
    print(f"{lang}: vocab size = {tokenizer.get_vocab_size()}")


def main() -> None:
    for lang in ("en", "ur"):
        train_tokenizer(lang)


if __name__ == "__main__":
    main()
