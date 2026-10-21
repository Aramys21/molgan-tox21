import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "deepchem",
        "torch",
        "rdkit",
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "pytest",
    ],
)
def test_required_package_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_deepchem_molgan_backend_imports():
    from deepchem.models.torch_models import BasicMolGANModel

    assert BasicMolGANModel is not None


def test_rdkit_parses_a_smiles():
    from rdkit import Chem

    molecule = Chem.MolFromSmiles("CCO")

    assert molecule is not None
    assert molecule.GetNumAtoms() == 3


def test_pytorch_runs_a_cpu_tensor_operation():
    import torch

    result = torch.tensor([1, 2, 3]).sum().item()

    assert result == 6
