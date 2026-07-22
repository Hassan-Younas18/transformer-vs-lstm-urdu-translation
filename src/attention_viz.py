"""Visualize the Transformer's decoder-to-encoder cross-attention: for a
handful of test sentences, render a heatmap of which English source tokens
the model attended to while generating each Urdu target token.
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams["font.family"] = ["Tahoma", "Segoe UI", "Noto Nastaliq Urdu", "Noto Sans Arabic", "Arial", "sans-serif"]

from dataset import TranslationDataset, load_tokenizers
from transformer_model import Transformer
from utils import CHECKPOINT_DIR, EOS_ID, MAX_LEN, PAD_ID, RESULTS_DIR, SOS_ID, get_device, shape_urdu


def plot_attention(src_tokens: list[str], tgt_tokens: list[str], attn: np.ndarray, out_path):
    # Arabic-script tokens need contextual reshaping + bidi reordering; matplotlib
    # (unlike a browser or a proper LaTeX RTL setup) doesn't shape text itself.
    shaped_tgt_tokens = [shape_urdu(t) for t in tgt_tokens]

    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.6), max(4, len(tgt_tokens) * 0.5)))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(shaped_tgt_tokens, fontsize=11)
    ax.set_xlabel("English source tokens")
    ax.set_ylabel("Urdu generated tokens")
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def generate_with_full_attention(model, src_tensor, sos_id, eos_id, max_len):
    """Greedy decode while recording the cross-attention row produced at
    every step, so we get a full (tgt_len x src_len) matrix rather than just
    the final step's attention.
    """
    device = src_tensor.device
    src_mask = model.make_src_mask(src_tensor)
    memory = model.encode(src_tensor, src_mask)
    ys = torch.full((1, 1), sos_id, dtype=torch.long, device=device)
    attn_rows = []
    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        dec_out, cross_attn = model.decode(ys, memory, src_mask, tgt_mask)
        # cross_attn: (B, n_heads, T, S) -> average heads, take last query position
        last_step_attn = cross_attn[0, :, -1, :].mean(dim=0)  # (S,)
        attn_rows.append(last_step_attn.cpu().numpy())
        logits = model.generator(dec_out[:, -1])
        next_token = logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == eos_id:
            break
    return ys.squeeze(0), np.stack(attn_rows)


def main(n_examples: int = 6):
    device = get_device()
    en_tok, ur_tok = load_tokenizers()
    test_ds = TranslationDataset("test", en_tok, ur_tok)

    model = Transformer(en_tok.get_vocab_size(), ur_tok.get_vocab_size(), max_len=MAX_LEN)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "transformer_best.pt", map_location=device))
    model.to(device).eval()

    out_dir = RESULTS_DIR / "attention_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(n_examples, len(test_ds))):
        src_tensor, _ = test_ds[i]
        src_ids = [t for t in src_tensor.tolist() if t not in (SOS_ID, EOS_ID, PAD_ID)]
        src_tokens = [en_tok.decode([tid]).strip() or "?" for tid in src_ids]

        src_input = src_tensor.unsqueeze(0).to(device)
        gen_ids, attn_matrix = generate_with_full_attention(model, src_input, SOS_ID, EOS_ID, MAX_LEN)
        gen_ids_list = [t for t in gen_ids.tolist() if t not in (SOS_ID, EOS_ID, PAD_ID)]
        tgt_tokens = [ur_tok.decode([tid]).strip() or "?" for tid in gen_ids_list]

        # attn_matrix rows = one per generated step; columns = full src_tensor incl. <sos>/<eos>.
        # Keep only real (non-EOS) target rows and drop the <sos>/<eos> source columns.
        attn_matrix = attn_matrix[: len(tgt_tokens), 1 : 1 + len(src_tokens)]

        plot_attention(src_tokens, tgt_tokens, attn_matrix, out_dir / f"example_{i}.png")
        print(f"saved {out_dir / f'example_{i}.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()
    main(args.n)
