"""A from-scratch implementation of the Transformer encoder-decoder
(Vaswani et al., 2017) for sequence-to-sequence translation.

Everything here — scaled dot-product attention, multi-head attention,
sinusoidal positional encoding, the position-wise feed-forward block, and
the encoder/decoder stacks — is implemented directly with tensor ops rather
than `nn.Transformer`, per the assignment requirement to build the
architecture from scratch.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import PAD_ID


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position embeddings, added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention, computed in parallel across `n_heads`."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        b = query.size(0)
        q = self.w_q(query).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, -1, self.n_heads * self.d_k)
        return self.w_o(out), attn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        cross_out, cross_attn = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x, cross_attn


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 80,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    @staticmethod
    def make_src_mask(src: torch.Tensor) -> torch.Tensor:
        return (src != PAD_ID).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

    @staticmethod
    def make_tgt_mask(tgt: torch.Tensor) -> torch.Tensor:
        pad_mask = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).bool()
        return pad_mask & causal_mask  # (B, 1, T, T)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.dropout(self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model)))
        cross_attn = None
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, memory, src_mask, tgt_mask)
        return x, cross_attn

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        dec_out, cross_attn = self.decode(tgt, memory, src_mask, tgt_mask)
        logits = self.generator(dec_out)
        return logits, cross_attn

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, sos_id: int, eos_id: int, max_len: int = 80):
        """Autoregressive greedy decoding, one sentence (batch size 1) at a time.
        Returns the generated token ids and the final step's cross-attention weights.
        """
        self.eval()
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)
        ys = torch.full((1, 1), sos_id, dtype=torch.long, device=src.device)
        cross_attn = None
        for _ in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(ys)
            dec_out, cross_attn = self.decode(ys, memory, src_mask, tgt_mask)
            logits = self.generator(dec_out[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == eos_id:
                break
        return ys.squeeze(0), cross_attn
