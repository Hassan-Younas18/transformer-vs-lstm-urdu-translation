# English → Urdu Neural Machine Translation (Assignment 3, Q1)

A from-scratch Transformer and an LSTM seq2seq baseline trained on the UMC005
English-Urdu parallel corpus, compared against a fine-tuned pretrained
MarianMT model, with BLEU/ROUGE evaluation, attention visualization, and a
ChatGPT-style Streamlit GUI.

**Results on the held-out test set:** Transformer 3.87 BLEU, LSTM 5.45 BLEU,
fine-tuned `Helsinki-NLP/opus-mt-en-ur` 22.50 BLEU. See `report/report.tex`
for the full writeup, methodology, and discussion.

## Project layout

```
data/                    raw + cleaned/split UMC005 corpus
artifacts/tokenizers/    trained BPE tokenizers (en, ur)
artifacts/checkpoints/   saved model weights (transformer_best.pt, lstm_best.pt)
src/                      data prep, models, training, evaluation, viz
app/streamlit_app.py      the GUI
results/                  loss curves, metrics, sample translations, attention plots
report/report.tex         IEEE conference-format technical report
report/build_sample_table.py   generates report/figures/sample_translations.png
kaggle/umc005_nmt_training.ipynb  self-contained notebook to (re)train on a Kaggle GPU
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

**Windows note:** if your project path is deeply nested (e.g. under a
OneDrive-synced folder) `pip install torch` can fail with a "filename too
long" error, since some of torch's own packaged file paths are long enough
to blow past Windows' 260-character `MAX_PATH` limit once combined with a
deep project path. If that happens, create the venv somewhere shallow
instead (e.g. `python -m venv C:\venvs\a3nmt`) and use that interpreter.

## Reproducing the pipeline

**Recommended: Kaggle GPU.** Local CPU training is slow (Transformer:
~30s/epoch got us to ~9 min total; a plain laptop CPU took closer to
19 min/epoch) and this project's local GPU (a 2GB MX350) failed to
initialize under PyTorch's CUDA build. `kaggle/umc005_nmt_training.ipynb`
is self-contained (embeds every `src/*.py` file via `%%writefile` cells) and
runs the whole pipeline on a free Kaggle T4 in well under an hour. Upload it
to kaggle.com, set Accelerator = GPU and Internet = On, Run All, then
download the zipped output (see the notebook's last cell) and extract it
into this project's `artifacts/` and `results/` folders.

**Local / CPU**, from the `src/` directory:

```bash
python data_prep.py                          # download + clean + split UMC005
python tokenizer_train.py                     # train BPE tokenizers (en, ur)
python train.py --model transformer --epochs 25
python train.py --model lstm --epochs 25
python evaluate.py --model transformer
python evaluate.py --model lstm
python attention_viz.py --n 6
python finetune_pretrained.py                  # bonus: fine-tune Helsinki-NLP/opus-mt-en-ur
```

Each `train.py` run early-stops on validation loss and writes:
- `artifacts/checkpoints/<model>_best.pt`
- `results/<model>_log.csv` (per-epoch train/val loss, used for the loss-curve plot)
- `results/<model>_meta.json` (params, training time, peak memory)

Each `evaluate.py` run writes `results/<model>_eval.json` (BLEU, ROUGE-1/2/L,
perplexity, inference speed) and `results/<model>_samples.json` (sample
translations). Note: `rouge_score`'s default tokenizer only matches
`[a-z0-9]` and silently zeroes every ROUGE score on non-Latin scripts —
`evaluate.py` and `finetune_pretrained.py` both pass `utils.WhitespaceTokenizer`
to `RougeScorer` explicitly to avoid this.

## Running the GUI

```bash
streamlit run app/streamlit_app.py
```

English input is left-aligned; the Urdu translation appears right-aligned
(RTL) below it. Conversation history persists for the session. The sidebar
lets you switch between the Transformer and LSTM checkpoints and toggle an
attention-heatmap view for the last translation.

## Regenerating report figures

```bash
python src/attention_viz.py --n 6              # results/attention_examples/*.png
python report/build_sample_table.py             # report/figures/sample_translations.png
```

Both rely on `arabic-reshaper` + `python-bidi` to properly shape/reorder
Urdu text for matplotlib/PIL, which don't do Arabic contextual letter-shaping
on their own (unlike a browser, which is why the Streamlit GUI doesn't need
this). The report embeds Urdu text as images for the same reason, so
`report.tex` compiles with plain `pdflatex` in Overleaf — no XeLaTeX/font
toolchain switch required.

## Submission packaging

Per the assignment instructions, zip this whole folder (code, report, and a
compiled/PDF or `.tex` of the report) as `RollNo_Name_Ass1.ZIP` before
submitting. Remember to fill in your actual name/roll number in
`report/report.tex`'s `\author{}` block first.
