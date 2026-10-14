"""
The purpose of this notebook is to pre-process the architectural floor plan drawings in `DATA_DIR`.
All outputs should be placed in `OUT_DIR`.

There should be separate files for the following items:
* Title Block (in its separate .pdf file)
* Measurements (in .yaml format with coordinates & distance)
* A .yaml file with room labels
* Drawing area (in its separate .pdf) with all room labels and measurement lines removed
* A .yaml and .pdf file with the legend
"""

from pathlib import Path

import pymupdf

from data_eng.parser import FloorPlanParser


def main() -> None:
    DATA_DIR = Path("tmp_dataset")
    OUT_DIR = Path("tmp_out")

    pdfs = []
    for entry in sorted(DATA_DIR.iterdir()):
        if entry.suffix.lower() == ".pdf":
            pdfs.append(entry)
        else:
            print(f"skipping non-PDF {entry}")
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {DATA_DIR}")

    for pdf in pdfs:
        doc = pymupdf.open(pdf)
        for page in doc:
            out_dir = OUT_DIR / pdf.stem / f"p{page.number:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            parser = FloorPlanParser(page)
            parser.parse()
            parser.get_measurements()
            parser.get_tags()
            parser.draw(out_dir)


if __name__ == "__main__":
    main()
