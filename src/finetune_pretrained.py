"""Bonus task: fine-tune the pretrained Helsinki-NLP/opus-mt-en-ur MarianMT
model on the same UMC005 train split, then evaluate it with the same
BLEU/ROUGE/perplexity metrics as the from-scratch models for a direct
custom-vs-pretrained comparison.
"""
import json
import math
import time

import psutil
import sacrebleu
import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from utils import CHECKPOINT_DIR, PROCESSED_DIR, RESULTS_DIR, WhitespaceTokenizer, get_device, read_lines

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"
OUT_DIR = CHECKPOINT_DIR / "finetuned_opus_mt"


def load_split(split: str) -> Dataset:
    en = read_lines(PROCESSED_DIR / f"{split}.en")
    ur = read_lines(PROCESSED_DIR / f"{split}.ur")
    return Dataset.from_dict({"en": en, "ur": ur})


def main(epochs: int = 3, batch_size: int = 16, max_len: int = 80):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    train_ds, dev_ds, test_ds = load_split("train"), load_split("dev"), load_split("test")

    def preprocess(batch):
        model_inputs = tokenizer(batch["en"], max_length=max_len, truncation=True)
        labels = tokenizer(text_target=batch["ur"], max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["en", "ur"])
    dev_tok = dev_ds.map(preprocess, batched=True, remove_columns=["en", "ur"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=False,
        logging_steps=50,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        data_collator=collator,
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train()
    training_time = time.time() - start
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    # --- Evaluation on the held-out test split ---
    model.eval()
    hyps, refs = [], []
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6
    infer_start = time.time()
    with torch.no_grad():
        for i in range(0, len(test_ds), batch_size):
            batch = test_ds[i : i + batch_size]
            inputs = tokenizer(batch["en"], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            generated = model.generate(**inputs, max_length=max_len)
            hyps.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            refs.extend(batch["ur"])
    infer_elapsed = time.time() - infer_start
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

    eval_loss = trainer.evaluate()["eval_loss"]
    perplexity = math.exp(min(eval_loss, 20))
    n_params = sum(p.numel() for p in model.parameters())

    results = {
        "model": "finetuned_opus_mt",
        "bleu": bleu.score,
        "rouge1_f": rouge_avg["rouge1"],
        "rouge2_f": rouge_avg["rouge2"],
        "rougeL_f": rouge_avg["rougeL"],
        "perplexity": perplexity,
        "test_loss": eval_loss,
        "n_params": n_params,
        "inference_sentences_per_sec": len(test_ds) / infer_elapsed,
        "inference_mem_delta_mb": mem_after - mem_before,
        "training_time_sec": training_time,
        "device": str(device),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "finetuned_opus_mt_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    samples = [{"src": s, "ref": r, "hyp": h} for s, r, h in list(zip(test_ds["en"], refs, hyps))[:20]]
    with open(RESULTS_DIR / "finetuned_opus_mt_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
