"""PyTorch Dataset + collate function for the English-Urdu parallel corpus."""
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import EOS_ID, MAX_LEN, PAD_ID, PROCESSED_DIR, SOS_ID, TOKENIZER_DIR, read_lines


def load_tokenizers() -> tuple[Tokenizer, Tokenizer]:
    en_tok = Tokenizer.from_file(str(TOKENIZER_DIR / "en.json"))
    ur_tok = Tokenizer.from_file(str(TOKENIZER_DIR / "ur.json"))
    return en_tok, ur_tok


class TranslationDataset(Dataset):
    """Tokenizes an (en, ur) split on the fly and wraps each side with <sos>/<eos>."""

    def __init__(self, split: str, en_tok: Tokenizer, ur_tok: Tokenizer, max_len: int = MAX_LEN):
        self.en_lines = read_lines(PROCESSED_DIR / f"{split}.en")
        self.ur_lines = read_lines(PROCESSED_DIR / f"{split}.ur")
        assert len(self.en_lines) == len(self.ur_lines)
        self.en_tok, self.ur_tok = en_tok, ur_tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.en_lines)

    def _encode(self, tok: Tokenizer, text: str) -> list[int]:
        ids = tok.encode(text).ids[: self.max_len - 2]
        return [SOS_ID] + ids + [EOS_ID]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src = self._encode(self.en_tok, self.en_lines[idx])
        tgt = self._encode(self.ur_tok, self.ur_lines[idx])
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
    return {"src": src_padded, "tgt": tgt_padded}
