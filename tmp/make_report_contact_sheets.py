from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
pages_dir = ROOT / "tmp" / "pdfs" / "rendered"
output_dir = ROOT / "tmp" / "pdfs" / "contact-sheets"
output_dir.mkdir(parents=True, exist_ok=True)

page_paths = sorted(pages_dir.glob("page-*.png"))
columns, rows = 4, 3
thumb_width = 300
label_height = 28
margin = 12

with Image.open(page_paths[0]) as sample:
    thumb_height = round(sample.height * thumb_width / sample.width)

batch_size = columns * rows
for sheet_index, start in enumerate(range(0, len(page_paths), batch_size), start=1):
    batch = page_paths[start : start + batch_size]
    width = margin + columns * (thumb_width + margin)
    height = margin + rows * (thumb_height + label_height + margin)
    sheet = Image.new("RGB", (width, height), "#D7DBE2")
    draw = ImageDraw.Draw(sheet)
    for offset, path in enumerate(batch):
        row, column = divmod(offset, columns)
        x = margin + column * (thumb_width + margin)
        y = margin + row * (thumb_height + label_height + margin)
        with Image.open(path) as page:
            thumbnail = page.copy()
            thumbnail.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, (x, y + label_height))
        draw.text((x + 4, y + 5), f"Page {start + offset + 1}", fill="#111827")
    sheet.save(output_dir / f"sheet-{sheet_index:02d}.jpg", quality=88)

print(f"sheets={sheet_index}")
