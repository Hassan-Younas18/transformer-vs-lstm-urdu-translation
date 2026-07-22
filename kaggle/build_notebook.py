"""Assemble a self-contained Kaggle notebook that reproduces the full
training pipeline (data prep -> tokenizers -> Transformer & LSTM training ->
evaluation -> attention viz -> bonus fine-tune) on a Kaggle GPU.

Each project source file is embedded verbatim via a `%%writefile` cell so the
notebook always matches whatever is currently in `../src/`. Run this script
locally after editing any src file to regenerate the notebook.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = Path(__file__).resolve().parent / "umc005_nmt_training.ipynb"

SRC_FILES = [
    "utils.py",
    "data_prep.py",
    "tokenizer_train.py",
    "dataset.py",
    "transformer_model.py",
    "lstm_model.py",
    "train.py",
    "evaluate.py",
    "attention_viz.py",
    "finetune_pretrained.py",
]

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# English -> Urdu NMT: Transformer vs. LSTM vs. Fine-tuned MarianMT\n"
    "Self-contained training pipeline for Assignment 3, Question 1 (UMC005 corpus).\n\n"
    "**Before running:** in the notebook's right-hand Settings panel, set\n"
    "**Accelerator = GPU (T4 x2 or P100)** and **Internet = On**, then Run All.\n\n"
    "At the end, `outputs.zip` will appear under the notebook's Output/Data tab —\n"
    "download it and unzip `artifacts/` and `results/` into your local project\n"
    "folder (next to `src/`) to use the trained checkpoints in the Streamlit GUI\n"
    "and to pull real numbers into the report."
))

cells.append(nbf.v4.new_code_cell(
    "!pip install -q tokenizers sacrebleu rouge-score sentencepiece"
))

cells.append(nbf.v4.new_code_cell(
    "import os\n"
    "os.makedirs('src', exist_ok=True)\n"
))

for fname in SRC_FILES:
    content = (SRC / fname).read_text(encoding="utf-8")
    cells.append(nbf.v4.new_code_cell(f"%%writefile src/{fname}\n{content}"))

cells.append(nbf.v4.new_markdown_cell("## 1. Data preparation"))
cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "sys.path.insert(0, 'src')\n"
    "import data_prep\n"
    "data_prep.main()"
))

cells.append(nbf.v4.new_markdown_cell("## 2. Train BPE tokenizers"))
cells.append(nbf.v4.new_code_cell(
    "import tokenizer_train\n"
    "tokenizer_train.main()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. Train the from-scratch Transformer\n"
    "GPU makes larger batches cheap, so we bump batch size vs. the CPU config."
))
cells.append(nbf.v4.new_code_cell(
    "import importlib\n"
    "import train\n"
    "importlib.reload(train)\n"
    "transformer_meta = train.train_model('transformer', epochs=30, batch_size=128, patience=5)"
))

cells.append(nbf.v4.new_markdown_cell("## 4. Train the LSTM baseline"))
cells.append(nbf.v4.new_code_cell(
    "lstm_meta = train.train_model('lstm', epochs=30, batch_size=128, patience=5)"
))

cells.append(nbf.v4.new_markdown_cell("## 5. Evaluate both models (BLEU / ROUGE / perplexity / speed)"))
cells.append(nbf.v4.new_code_cell(
    "import evaluate\n"
    "transformer_results = evaluate.evaluate_model('transformer')\n"
    "lstm_results = evaluate.evaluate_model('lstm')"
))

cells.append(nbf.v4.new_markdown_cell("## 6. Attention visualization"))
cells.append(nbf.v4.new_code_cell(
    "import attention_viz\n"
    "attention_viz.main(n_examples=6)"
))

cells.append(nbf.v4.new_markdown_cell("## 7. Loss curve plot (train vs. val, both models)"))
cells.append(nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n\n"
    "fig, ax = plt.subplots(figsize=(7, 5))\n"
    "for name, color in [('transformer', 'tab:blue'), ('lstm', 'tab:orange')]:\n"
    "    df = pd.read_csv(f'results/{name}_log.csv')\n"
    "    ax.plot(df['epoch'], df['train_loss'], color=color, linestyle='-', label=f'{name} train')\n"
    "    ax.plot(df['epoch'], df['val_loss'], color=color, linestyle='--', label=f'{name} val')\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.set_title('Training / Validation Loss')\n"
    "fig.tight_layout()\n"
    "fig.savefig('results/loss_curves.png', dpi=150)\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 8. Bonus: fine-tune the pretrained MarianMT model\n"
    "Fine-tunes `Helsinki-NLP/opus-mt-en-ur` on the same train split for comparison."
))
cells.append(nbf.v4.new_code_cell(
    "!pip install -q transformers datasets accelerate\n"
    "import finetune_pretrained\n"
    "finetune_pretrained.main()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 9. Package everything for download\n"
    "We deliberately **exclude** the fine-tuned MarianMT checkpoint's raw weights here --\n"
    "with optimizer state it can be 500MB-1GB and isn't needed locally (the GUI only\n"
    "serves the two from-scratch models; the bonus model's *metrics* are all the report\n"
    "needs, and those live in `results/`). This keeps the zip small and the download reliable."
))
cells.append(nbf.v4.new_code_cell(
    "import zipfile\n"
    "from pathlib import Path\n\n"
    "INCLUDE = [\n"
    "    'artifacts/checkpoints/transformer_best.pt',\n"
    "    'artifacts/checkpoints/lstm_best.pt',\n"
    "]\n"
    "with zipfile.ZipFile('/kaggle/working/outputs.zip', 'w', zipfile.ZIP_DEFLATED) as z:\n"
    "    for f in INCLUDE:\n"
    "        if Path(f).exists():\n"
    "            z.write(f, arcname=f)\n"
    "    for pattern_dir in ['artifacts/tokenizers', 'results']:\n"
    "        for p in Path(pattern_dir).rglob('*'):\n"
    "            if p.is_file():\n"
    "                z.write(p, arcname=p)\n\n"
    "size_mb = Path('/kaggle/working/outputs.zip').stat().st_size / 1e6\n"
    "print(f'Wrote /kaggle/working/outputs.zip ({size_mb:.1f} MB).')\n"
    "print('If the Output pane download does not work, use Save Version -> Save & Run All (Commit),\\n'\n"
    "      'then open the committed version and download outputs.zip from its Output tab.')"
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "accelerator": "GPU",
}

nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
