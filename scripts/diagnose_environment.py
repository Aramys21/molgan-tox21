"""Report interpreter/package versions, run import smoke checks, and inspect CUDA."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys


PACKAGES = {
    "deepchem": "deepchem",
    "torch": "torch",
    "rdkit": "rdkit",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
}


def inspect_environment() -> dict:
    report = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "architecture": platform.architecture()[0],
        "packages": {},
        "imports": {},
        "cuda": {
            "available": False,
            "torch_build": None,
            "devices": [],
            "nvidia_smi": None,
        },
    }

    for distribution, module_name in PACKAGES.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        report["packages"][distribution] = version
        try:
            importlib.import_module(module_name)
            report["imports"][module_name] = "PASS"
        except Exception as error:
            report["imports"][module_name] = f"FAIL: {type(error).__name__}: {error}"

    try:
        import deepchem as dc
        from deepchem.models.torch_models import BasicMolGANModel

        report["imports"]["deepchem.models.torch_models.BasicMolGANModel"] = (
            "PASS" if BasicMolGANModel else "FAIL: import returned no class"
        )
        report["deepchem_version"] = dc.__version__
    except Exception as error:
        report["imports"]["deepchem.models.torch_models.BasicMolGANModel"] = (
            f"FAIL: {type(error).__name__}: {error}"
        )

    try:
        import torch

        report["cuda"]["torch_build"] = torch.version.cuda
        report["cuda"]["available"] = bool(torch.cuda.is_available())
        if report["cuda"]["available"]:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                report["cuda"]["devices"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "compute_capability": (
                            f"{properties.major}.{properties.minor}"
                        ),
                        "memory_gib": round(
                            properties.total_memory / (1024**3),
                            2,
                        ),
                    }
                )
    except Exception as error:
        report["cuda"]["torch_error"] = f"{type(error).__name__}: {error}"

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            report["cuda"]["nvidia_smi"] = (
                result.stdout.strip()
                if result.returncode == 0
                else result.stderr.strip()
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            report["cuda"]["nvidia_smi"] = f"{type(error).__name__}: {error}"
    return report


def print_report(report: dict) -> None:
    print(f"OS: {report['os']}")
    print(
        f"Python: {report['python']} ({report['architecture']})"
        f"\nInterpréteur: {report['python_executable']}"
    )
    print("\nVersions des paquets:")
    for package, version in report["packages"].items():
        print(f"  {package}: {version or 'ABSENT'}")
    print("\nImport smoke tests:")
    for module, status in report["imports"].items():
        print(f"  {module}: {status}")
    print(
        "\nCUDA: "
        f"PyTorch build={report['cuda']['torch_build']}, "
        f"available={report['cuda']['available']}"
    )
    for device in report["cuda"]["devices"]:
        print(
            f"  GPU {device['index']}: {device['name']} "
            f"(compute {device['compute_capability']}, "
            f"{device['memory_gib']} GiB)"
        )
    if report["cuda"]["nvidia_smi"]:
        print(f"  nvidia-smi: {report['cuda']['nvidia_smi']}")


def main() -> int:
    report = inspect_environment()
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0 if all(
        status == "PASS" for status in report["imports"].values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
