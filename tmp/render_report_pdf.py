from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp" / "pymupdf"))
import fitz


pdf_path = ROOT / "tmp" / "pdfs" / "report.pdf"
output_dir = ROOT / "tmp" / "pdfs" / "rendered"
output_dir.mkdir(parents=True, exist_ok=True)

document = fitz.open(pdf_path)
matrix = fitz.Matrix(1.55, 1.55)
for index, page in enumerate(document):
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    pixmap.save(output_dir / f"page-{index + 1:03d}.png")
print(f"pages={document.page_count}")
print(f"output={output_dir}")
