from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tox21 import MOLNET_AUDIT_FILE, MOLNET_CACHE_DIR, load_tox21_splits, write_tox21_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Charge et audite les partitions officielles Tox21 DeepChem."
    )
    parser.add_argument("--cache-dir", type=Path, default=MOLNET_CACHE_DIR)
    parser.add_argument("--audit-file", type=Path, default=MOLNET_AUDIT_FILE)
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Force DeepChem à refaire la featurization et les partitions.",
    )
    args = parser.parse_args()

    splits = load_tox21_splits(
        cache_dir=args.cache_dir,
        featurizer="GraphConv",
        splitter="scaffold",
        reload=not args.no_reload,
    )
    report = write_tox21_audit(splits, args.audit_file)
    print(
        f"DeepChem {report['deepchem_version']} | tâches : "
        f"{', '.join(report['tasks'])}"
    )
    print(
        f"Partitions officielles scaffold: "
        f"train={len(splits.train)}, valid={len(splits.valid)}, "
        f"test={len(splits.test)}"
    )
    print(
        f"Features : {report['feature_object_type']} | "
        f"IDs SMILES : {report['ids']['contains_smiles']}"
    )
    print(f"Rapport d'audit : {args.audit_file}")
    for split_name, stats in report["splits"].items():
        print(
            f"{split_name}: {stats['compounds']} composés, "
            f"{stats['observed_label_cells']} labels observés, "
            f"{stats['masked_label_cells']} masqués"
        )
    print(json.dumps(report["splits"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
