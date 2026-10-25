"""Create image grids for generated RDKit molecules."""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw


def save_molecule_grid(molecules, output_file: Path, legends=None) -> Path:
    molecules = list(molecules)
    if not molecules:
        raise ValueError("Impossible de dessiner une liste vide de molécules.")
    sanitized_molecules = []
    for index, molecule in enumerate(molecules):
        if molecule is None:
            raise ValueError(f"La molécule à l'index {index} est absente.")
        try:
            sanitized = Chem.Mol(molecule)
            Chem.SanitizeMol(sanitized)
        except Exception as error:
            raise ValueError(
                f"La molécule à l'index {index} n'est pas chimiquement valide : "
                f"{type(error).__name__}: {error}"
            ) from error
        sanitized_molecules.append(sanitized)

    if legends is not None and len(legends) != len(sanitized_molecules):
        raise ValueError("Le nombre de légendes doit correspondre aux molécules.")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image = Draw.MolsToGridImage(
        sanitized_molecules,
        legends=legends,
        molsPerRow=4,
        subImgSize=(300, 240),
        useSVG=False,
        returnPNG=True,
    )
    png_data = image if isinstance(image, bytes) else getattr(image, "data", None)
    if not isinstance(png_data, bytes):
        raise TypeError(
            "RDKit n'a pas retourné les octets PNG attendus pour la grille moléculaire."
        )
    output_file.write_bytes(png_data)
    return output_file
