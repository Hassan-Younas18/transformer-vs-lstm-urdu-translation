"""ChatGPT-style GUI for the English->Urdu translators.

English input is left-aligned (LTR); the Urdu translation appears below it,
right-aligned and rendered right-to-left. Conversation history persists for
the session, and a sidebar lets the user pick which trained model serves the
translation and optionally inspect the attention heatmap for the last turn.
"""
import sys
from pathlib import Path

import streamlit as st
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from attention_viz import generate_with_full_attention, plot_attention  # noqa: E402
from dataset import load_tokenizers  # noqa: E402
from lstm_model import Seq2SeqLSTM  # noqa: E402
from transformer_model import Transformer  # noqa: E402
from utils import CHECKPOINT_DIR, EOS_ID, MAX_LEN, PAD_ID, SOS_ID, get_device  # noqa: E402

st.set_page_config(page_title="English -> Urdu Translator", page_icon="🌐", layout="centered")

CUSTOM_CSS = """
<style>
.chat-row { display: flex; flex-direction: column; margin-bottom: 1.1rem; }
.bubble-en {
    align-self: flex-start; background: #2f6fed; color: white;
    padding: 0.6rem 1rem; border-radius: 14px 14px 14px 2px;
    max-width: 75%; font-size: 1rem; direction: ltr; text-align: left;
}
.bubble-ur {
    align-self: flex-end; background: #edf1f7; color: #111;
    padding: 0.6rem 1rem; border-radius: 14px 14px 2px 14px;
    max-width: 75%; font-size: 1.15rem; direction: rtl; text-align: right;
    margin-top: 0.35rem; font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
}
@media (prefers-color-scheme: dark) {
    .bubble-ur { background: #2a2d34; color: #f2f2f2; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    "<link href='https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap' rel='stylesheet'>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_name: str):
    device = get_device()
    en_tok, ur_tok = load_tokenizers()
    if model_name == "transformer":
        model = Transformer(en_tok.get_vocab_size(), ur_tok.get_vocab_size(), max_len=MAX_LEN)
        ckpt = CHECKPOINT_DIR / "transformer_best.pt"
    else:
        model = Seq2SeqLSTM(en_tok.get_vocab_size(), ur_tok.get_vocab_size())
        ckpt = CHECKPOINT_DIR / "lstm_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model, en_tok, ur_tok, device


def translate(model, en_tok, ur_tok, device, text: str, model_name: str, want_attention: bool):
    ids = [SOS_ID] + en_tok.encode(text).ids[: MAX_LEN - 2] + [EOS_ID]
    src_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    if model_name == "transformer" and want_attention:
        gen_ids, attn = generate_with_full_attention(model, src_tensor, SOS_ID, EOS_ID, MAX_LEN)
    else:
        gen_ids, attn = model.greedy_decode(src_tensor, SOS_ID, EOS_ID, MAX_LEN)

    out_ids = [t for t in gen_ids.tolist() if t not in (SOS_ID, EOS_ID, PAD_ID)]
    translation = ur_tok.decode(out_ids)

    fig_path = None
    if model_name == "transformer" and want_attention and len(out_ids) > 0:
        src_ids = [t for t in ids if t not in (SOS_ID, EOS_ID, PAD_ID)]
        src_tokens = [en_tok.decode([tid]).strip() or "?" for tid in src_ids]
        tgt_tokens = [ur_tok.decode([tid]).strip() or "?" for tid in out_ids]
        attn_matrix = attn[: len(tgt_tokens), 1 : 1 + len(src_tokens)]
        fig_path = Path("results") / "attention_examples" / "_gui_last.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_attention(src_tokens, tgt_tokens, attn_matrix, fig_path)

    return translation, fig_path


st.title("🌐 English → Urdu Translator")
st.caption("From-scratch Transformer & LSTM models trained on the UMC005 English-Urdu corpus")

available_models = []
if (CHECKPOINT_DIR / "transformer_best.pt").exists():
    available_models.append("transformer")
if (CHECKPOINT_DIR / "lstm_best.pt").exists():
    available_models.append("lstm")

if not available_models:
    st.error("No trained checkpoints found in artifacts/checkpoints/. Run `src/train.py` first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", available_models, format_func=str.title)
    show_attention = st.checkbox("Show attention heatmap (Transformer only)", value=False)
    if st.button("Clear conversation"):
        st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []

model, en_tok, ur_tok, device = load_model(model_choice)

for turn in st.session_state.history:
    st.markdown(
        f'<div class="chat-row"><div class="bubble-en">{turn["en"]}</div>'
        f'<div class="bubble-ur">{turn["ur"]}</div></div>',
        unsafe_allow_html=True,
    )
    if turn.get("attn_path") and Path(turn["attn_path"]).exists():
        with st.expander("Attention heatmap for this translation"):
            st.image(turn["attn_path"])

user_text = st.chat_input("Type an English sentence and press Enter...")
if user_text:
    with st.spinner("Translating..."):
        translation, attn_path = translate(
            model, en_tok, ur_tok, device, user_text, model_choice, show_attention
        )
    st.session_state.history.append(
        {"en": user_text, "ur": translation, "attn_path": str(attn_path) if attn_path else None}
    )
    st.rerun()
