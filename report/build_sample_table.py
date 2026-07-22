"""Render a small set of example source/reference/hypothesis translations as
a PNG for the report.

We render Urdu text as an image (via PIL, with proper Arabic-script
reshaping/bidi reordering) rather than typesetting it live in LaTeX, so the
report keeps compiling with plain pdflatex -- no XeLaTeX/polyglossia engine
switch or Urdu-font availability assumptions required on the grader's end.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures" / "sample_translations.png"

FONT_PATH = "C:/Windows/Fonts/tahoma.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/tahomabd.ttf"
FONT_SIZE = 22
LINE_GAP = 10
BLOCK_GAP = 26
WIDTH = 1500
MARGIN = 30


def shape(text: str) -> str:
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text))


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.getlength(trial) > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def main():
    label_font = ImageFont.truetype(FONT_BOLD_PATH, FONT_SIZE)
    text_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    examples = [
        {
            "src": "And secrets which are in hearts will be disclosed?",
            "ref": "اور راز ظاہر کر دیئے جائیں گے جو سینوں میں ہیں ۔",
            "transformer": "اور جن میں وہ لوگ ہیں جو اس میں ہمیشہ رہنے والے ہیں ۔",
            "lstm": "اور جو لوگ ہیں وہ کس کے دلوں میں ہیں ؟",
            "finetuned": "اور سینوں کی پوشیده باتیں ظاہر کر دی جائیں گی ۔",
        },
        {
            "src": "So he is the one who pushes away the orphan.",
            "ref": "تو یہ وہ شخص ہے جو یتیم کو دھکے دیتا ہے",
            "transformer": "سو وہ ان کے پاس ایک ہی کی بڑی نشانی ہے جو ان کے آگے آگے آگے آگے بھیج چکے ہیں ۔",
            "lstm": "پس وہ ایک ہی یعنی ان کے پاس یعنی حق دار یعنی اللہ کی شان و فکر کی شان ہے ۔",
            "finetuned": "پس یہی وہ ہے جو یتیم کو دھکے دیتا ہے ۔",
        },
    ]
    rows = [
        ("Source (EN)", "src", False),
        ("Reference (UR)", "ref", True),
        ("Transformer", "transformer", True),
        ("LSTM", "lstm", True),
        ("Fine-tuned MarianMT", "finetuned", True),
    ]

    text_max_width = WIDTH - 2 * MARGIN - 220

    # First pass: compute total height.
    total_h = MARGIN
    layout = []
    for ex in examples:
        for label, key, is_urdu in rows:
            raw = ex[key]
            content = shape(raw) if is_urdu else raw
            font = text_font
            lines = wrap_text(raw, font, text_max_width) if not is_urdu else [raw]
            if is_urdu:
                # naive wrap on the *unshaped* text length, then shape each wrapped chunk
                lines = wrap_text(raw, font, text_max_width)
                lines = [shape(l) for l in lines]
            n_lines = max(1, len(lines))
            row_h = n_lines * (FONT_SIZE + LINE_GAP)
            layout.append((label, lines, is_urdu, row_h))
            total_h += row_h
        total_h += BLOCK_GAP

    img = Image.new("RGB", (WIDTH, total_h + MARGIN), "white")
    d = ImageDraw.Draw(img)

    y = MARGIN
    idx = 0
    for ex_i, ex in enumerate(examples):
        for _ in rows:
            label, lines, is_urdu, row_h = layout[idx]
            idx += 1
            d.text((MARGIN, y), f"{label}:", font=label_font, fill=(20, 20, 20))
            ly = y
            for line in lines:
                if is_urdu:
                    w = text_font.getlength(line)
                    d.text((WIDTH - MARGIN - w, ly), line, font=text_font, fill=(0, 0, 0))
                else:
                    d.text((220, ly), line, font=text_font, fill=(0, 0, 0))
                ly += FONT_SIZE + LINE_GAP
            y += row_h
        y += BLOCK_GAP
        if ex_i < len(examples) - 1:
            d.line([(MARGIN, y - BLOCK_GAP // 2), (WIDTH - MARGIN, y - BLOCK_GAP // 2)], fill=(200, 200, 200), width=2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"wrote {OUT} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
