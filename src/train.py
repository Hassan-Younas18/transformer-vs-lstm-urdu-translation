"""Shared training loop for both the from-scratch Transformer and the LSTM
baseline: teacher forcing, LR scheduling, early stopping, gradient clipping,
checkpointing, and per-epoch CSV logging (used later for the loss-curve plot).
"""
import argparse
import csv
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TranslationDataset, collate_fn, load_tokenizers
from lstm_model import Seq2SeqLSTM
from transformer_model import Transformer
from utils import CHECKPOINT_DIR, MAX_LEN, PAD_ID, RESULTS_DIR, get_device, set_seed


class NoamScheduler:
    """Warmup + inverse-sqrt-decay LR schedule from 'Attention Is All You Need' (sec 5.3)."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model**-0.5 * min(self.step_num**-0.5, self.step_num * self.warmup_steps**-1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()


def run_epoch(model, loader, optimizer, criterion, device, scheduler=None, train=True):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        src, tgt = batch["src"].to(device), batch["tgt"].to(device)
        with torch.set_grad_enabled(train):
            if isinstance(model, Transformer):
                logits, _ = model(src, tgt[:, :-1])
            else:
                logits, _ = model(src, tgt, teacher_forcing_ratio=1.0 if train else 1.0)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scheduler is not None:
                scheduler.step()
            else:
                optimizer.step()
        n_tokens = (tgt[:, 1:] != PAD_ID).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def train_model(model_name: str, epochs: int, batch_size: int, patience: int = 5, lr: float = 3e-4):
    set_seed(42)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"Training {model_name} on device={device}")

    en_tok, ur_tok = load_tokenizers()
    train_ds = TranslationDataset("train", en_tok, ur_tok)
    dev_ds = TranslationDataset("dev", en_tok, ur_tok)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    src_vocab_size = en_tok.get_vocab_size()
    tgt_vocab_size = ur_tok.get_vocab_size()

    if model_name == "transformer":
        model = Transformer(src_vocab_size, tgt_vocab_size, max_len=MAX_LEN).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=model.d_model, warmup_steps=2000)
        plateau_scheduler = None
    elif model_name == "lstm":
        model = Seq2SeqLSTM(src_vocab_size, tgt_vocab_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    else:
        raise ValueError(model_name)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name}: {n_params:,} parameters")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"{model_name}_log.csv"
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()
    peak_mem_mb = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_sec"])

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss = run_epoch(model, train_loader, optimizer, criterion, device, scheduler, train=True)
            val_loss = run_epoch(model, dev_loader, optimizer, criterion, device, scheduler=None, train=False)

            if plateau_scheduler is not None:
                plateau_scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if device.type == "cuda":
                peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / 1e6)

            elapsed = time.time() - epoch_start
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{current_lr:.6f}", f"{elapsed:.1f}"])
            f.flush()
            print(
                f"[{model_name}] epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} lr={current_lr:.6f} ({elapsed:.1f}s)"
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), CHECKPOINT_DIR / f"{model_name}_best.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"[{model_name}] early stopping at epoch {epoch} (patience={patience})")
                    break

    total_time = time.time() - start_time
    meta = {
        "model": model_name,
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "training_time_sec": total_time,
        "peak_gpu_mem_mb": peak_mem_mb,
        "device": str(device),
        "epochs_run": epoch,
    }
    with open(RESULTS_DIR / f"{model_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{model_name}] done in {total_time:.1f}s, best_val_loss={best_val_loss:.4f}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "lstm"], required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    train_model(args.model, args.epochs, args.batch_size, args.patience)
