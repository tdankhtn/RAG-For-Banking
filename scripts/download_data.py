import argparse
from pathlib import Path

import gdown

from src.config import settings


DEFAULT_PDFS = [
    {
        "title": "cau_hoi_thuong_gap",
        "file_id": "1H83r8CT_nIvs3_SLPe317IfXUabzO5HS",
    },
    {
        "title": "bieu_phi_dich_vu_the_thanh_toan",
        "file_id": "148jkm6_n8wgn4_O1UHX2XBPrHcQI8iJC",
    },
    {
        "title": "bieu_phi_dich_vu_the_tin_dung",
        "file_id": "1WjC2CPF3hutist5ncd0JFBuBuEz5JSMP", 
    }
]


def download_pdf(file_id: str, output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"Skip (exists): {output_path}")
        return

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, output=str(output_path), quiet=False)


def main():
    parser = argparse.ArgumentParser(description="Download sample PDF data.")
    parser.add_argument("--force", action="store_true", help="Redownload files")
    args = parser.parse_args()

    for item in DEFAULT_PDFS:
        output_path = settings.data_dir / f"{item['title']}.pdf"
        download_pdf(item["file_id"], output_path, force=args.force)


if __name__ == "__main__":
    main()
