"""Create image grids for generated RDKit molecules."""

from pathlib import Path

from rdkit.Chem import Draw


def save_molecule_grid(molecules, output_file: Path, legends=None) -> Path:
    molecules = list(molecules)
    if not molecules:
        raise ValueError("Impossible de dessiner une liste vide de molécules.")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image = Draw.MolsToGridImage(
        molecules,
        legends=legends,
        molsPerRow=4,
        subImgSize=(300, 240),
        useSVG=False,
    )
    image.save(output_file)
    return output_file
