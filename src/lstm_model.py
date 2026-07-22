"""Seq2seq LSTM baseline with Luong (multiplicative) attention.

Architectural counterpart to the from-scratch Transformer: a bidirectional
LSTM encoder, an autoregressive LSTM decoder, and a Luong-style attention
mechanism over the encoder outputs at every decoding step. Used purely as a
comparison point in the report (training time, BLEU/ROUGE, perplexity, etc.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import PAD_ID


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        lengths = (src != PAD_ID).sum(dim=1).clamp(min=1).cpu()
        embedded = self.dropout(self.embed(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1)
        )
        # h, c: (2, B, hidden) from the single bidirectional layer -> merge into decoder's initial state
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        c_cat = torch.cat([c[0], c[1]], dim=-1)
        h0 = torch.tanh(self.bridge_h(h_cat)).unsqueeze(0)
        c0 = torch.tanh(self.bridge_c(c_cat)).unsqueeze(0)
        return outputs, (h0, c0)


class LuongAttention(nn.Module):
    """General (bilinear) attention: score(h_t, h_s) = h_t^T W h_s."""

    def __init__(self, hidden_dim: int, encoder_dim: int):
        super().__init__()
        self.W = nn.Linear(encoder_dim, hidden_dim, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor, src_mask: torch.Tensor):
        # dec_hidden: (B, hidden), enc_outputs: (B, S, encoder_dim)
        proj = self.W(enc_outputs)  # (B, S, hidden)
        scores = torch.bmm(proj, dec_hidden.unsqueeze(2)).squeeze(2)  # (B, S)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, S)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # (B, encoder_dim)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, encoder_dim: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.attention = LuongAttention(hidden_dim, encoder_dim)
        self.out = nn.Linear(hidden_dim + encoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden, enc_outputs, src_mask):
        embedded = self.dropout(self.embed(input_token))  # (B, 1, emb)
        dec_out, hidden = self.lstm(embedded, hidden)  # dec_out: (B, 1, hidden)
        context, attn_weights = self.attention(dec_out.squeeze(1), enc_outputs, src_mask)
        logits = self.out(torch.cat([dec_out.squeeze(1), context], dim=-1))
        return logits, hidden, attn_weights


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hidden_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, hidden_dim, hidden_dim * 2, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 1.0):
        src_mask = src != PAD_ID
        enc_outputs, hidden = self.encoder(src)

        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.out.out_features
        logits_all = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)

        input_token = tgt[:, :1]  # <sos>
        for t in range(tgt_len - 1):
            logits, hidden, _ = self.decoder.forward_step(input_token, hidden, enc_outputs, src_mask)
            logits_all[:, t] = logits
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            input_token = tgt[:, t + 1 : t + 2] if use_teacher else logits.argmax(-1, keepdim=True)
        return logits_all, None

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, sos_id: int, eos_id: int, max_len: int = 80):
        self.eval()
        src_mask = src != PAD_ID
        enc_outputs, hidden = self.encoder(src)
        input_token = torch.full((1, 1), sos_id, dtype=torch.long, device=src.device)
        tokens = [sos_id]
        attn_history = []
        for _ in range(max_len - 1):
            logits, hidden, attn_weights = self.decoder.forward_step(input_token, hidden, enc_outputs, src_mask)
            next_token = logits.argmax(-1, keepdim=True)
            tokens.append(next_token.item())
            attn_history.append(attn_weights.squeeze(0))
            if next_token.item() == eos_id:
                break
            input_token = next_token
        return torch.tensor(tokens, device=src.device), torch.stack(attn_history)
