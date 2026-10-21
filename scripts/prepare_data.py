from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tox21 import ANALYSIS_FILE, PROCESSED_DIR, RAW_FILE, analyze_and_prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Nettoie et analyse Tox21.")
    parser.add_argument("--input", type=Path, default=RAW_FILE)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    clean, report = analyze_and_prepare(args.input, args.output_dir)
    report_path = args.output_dir / ANALYSIS_FILE.name
    print(f"Lignes brutes : {report['raw_rows']}")
    print(f"Lignes nettoyées : {len(clean)}")
    print(
        "Molécules compatibles MolGAN : "
        f"{report['molgan_filter']['compatible_rows']}"
    )
    print(f"Rapport : {report_path}")
    print(json.dumps(report["tasks"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
