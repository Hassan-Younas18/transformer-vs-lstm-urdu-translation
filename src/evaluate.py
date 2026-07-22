"""Evaluate a trained model on the test set: BLEU, ROUGE-1/2/L, perplexity,
inference speed, and (for the two from-scratch models) parameter count /
training time / memory pulled from the training-run metadata JSON.
"""
import argparse
import json
import math
import time

import psutil
import sacrebleu
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader

from dataset import TranslationDataset, collate_fn, load_tokenizers
from lstm_model import Seq2SeqLSTM
from train import run_epoch
from transformer_model import Transformer
from utils import CHECKPOINT_DIR, EOS_ID, MAX_LEN, PAD_ID, RESULTS_DIR, SOS_ID, WhitespaceTokenizer, get_device


def decode_greedy(model, src_tensor, sos_id, eos_id, max_len):
    ids, attn = model.greedy_decode(src_tensor, sos_id, eos_id, max_len)
    return ids, attn


def evaluate_model(model_name: str, batch_size: int = 32, max_len: int = MAX_LEN):
    device = get_device()
    en_tok, ur_tok = load_tokenizers()
    test_ds = TranslationDataset("test", en_tok, ur_tok, max_len=max_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    src_vocab_size, tgt_vocab_size = en_tok.get_vocab_size(), ur_tok.get_vocab_size()
    if model_name == "transformer":
        model = Transformer(src_vocab_size, tgt_vocab_size, max_len=max_len)
    elif model_name == "lstm":
        model = Seq2SeqLSTM(src_vocab_size, tgt_vocab_size)
    else:
        raise ValueError(model_name)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{model_name}_best.pt", map_location=device))
    model.to(device).eval()

    # Perplexity from cross-entropy loss over the test set.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.0)
    test_loss = run_epoch(model, test_loader, optimizer=None, criterion=criterion, device=device, train=False)
    perplexity = math.exp(min(test_loss, 20))

    # Greedy decode sentence-by-sentence for BLEU/ROUGE + inference speed.
    hyps, refs, srcs = [], [], []
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6
    start = time.time()
    for i in range(len(test_ds)):
        src_tensor, tgt_tensor = test_ds[i]
        src_tensor = src_tensor.unsqueeze(0).to(device)
        ids, _ = decode_greedy(model, src_tensor, SOS_ID, EOS_ID, max_len)
        hyp_ids = [t for t in ids.tolist() if t not in (SOS_ID, EOS_ID, PAD_ID)]
        ref_ids = [t for t in tgt_tensor.tolist() if t not in (SOS_ID, EOS_ID, PAD_ID)]
        hyps.append(ur_tok.decode(hyp_ids))
        refs.append(ur_tok.decode(ref_ids))
        srcs.append(en_tok.decode([t for t in src_tensor.squeeze(0).tolist() if t not in (SOS_ID, EOS_ID, PAD_ID)]))
    elapsed = time.time() - start
    mem_after = process.memory_info().rss / 1e6

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=WhitespaceTokenizer()
    )
    rouge_sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for hyp, ref in zip(hyps, refs):
        scores = scorer.score(ref, hyp)
        for k in rouge_sums:
            rouge_sums[k] += scores[k].fmeasure
    rouge_avg = {k: v / len(hyps) for k, v in rouge_sums.items()}

    n_params = sum(p.numel() for p in model.parameters())
    meta_path = RESULTS_DIR / f"{model_name}_meta.json"
    train_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    results = {
        "model": model_name,
        "bleu": bleu.score,
        "rouge1_f": rouge_avg["rouge1"],
        "rouge2_f": rouge_avg["rouge2"],
        "rougeL_f": rouge_avg["rougeL"],
        "perplexity": perplexity,
        "test_loss": test_loss,
        "n_params": n_params,
        "inference_sentences_per_sec": len(test_ds) / elapsed,
        "inference_mem_delta_mb": mem_after - mem_before,
        "training_time_sec": train_meta.get("training_time_sec"),
        "peak_gpu_mem_mb": train_meta.get("peak_gpu_mem_mb"),
        "device": str(device),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{model_name}_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    samples = [{"src": s, "ref": r, "hyp": h} for s, r, h in list(zip(srcs, refs, hyps))[:20]]
    with open(RESULTS_DIR / f"{model_name}_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "lstm"], required=True)
    args = parser.parse_args()
    evaluate_model(args.model)
