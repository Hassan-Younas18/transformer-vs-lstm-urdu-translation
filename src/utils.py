"""Shared constants and small helpers used across the training/eval scripts."""
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TOKENIZER_DIR = ROOT / "artifacts" / "tokenizers"
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = ROOT / "results"

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

VOCAB_SIZE = 8000
MAX_LEN = 80


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch

    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return torch.device("cuda")
        except RuntimeError:
            return torch.device("cpu")
    return torch.device("cpu")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def shape_urdu(text: str) -> str:
    """Reshape + bidi-reorder Urdu (Arabic-script) text for renderers that
    don't perform contextual letter-shaping themselves (matplotlib, PIL).
    Browsers and LaTeX with proper RTL support don't need this.
    """
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text))


class WhitespaceTokenizer:
    """Simple whitespace tokenizer for `rouge_score.RougeScorer`.

    `rouge_score`'s default tokenizer matches only `[a-z0-9]`, which silently
    strips all Urdu (Arabic-script) text to an empty token list and forces
    every ROUGE score to 0 regardless of translation quality. Our corpus and
    tokenizer output are already whitespace-segmented (words and punctuation
    separated by spaces), so a plain `.split()` is a correct, script-agnostic
    substitute.
    """

    def tokenize(self, text: str) -> list[str]:
        return text.split()
