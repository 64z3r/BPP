import os
import logging

from typing import Iterable
from pathlib import Path
from rdkit import Chem

logger = logging.getLogger(__name__)


class MolFormatError(ValueError):
    """Raised if molecule file cannot be read."""


def read_mol(
    path: Path | str,
    sanitize: bool = False,
    removeHs: bool = False,
) -> Chem.rdchem.Mol:
    """Reads molecule file.

    Files containing supported formats end with `".mol2"`, `".sdf"`, `".pdb"`
    and `".pdbqt"`.

    Args:
        path: Path to molecule file.
        sanitize: Whether to sanitize the loaded molecule.
        removeHs: Whether to remove hydrogen atoms from the molecule.

    Returns:
        Loaded molecule file.

    Raises:
        MolFormatError: If molecule file isn't supported or couldn't be loaded.
    """

    path = Path(path)

    def process(mol):
        if mol is not None:
            mol.SetProp("Name", path.stem)
            return mol
        raise MolFormatError(f"cannot parse molecule file '{path!s}'")

    if path.suffix == ".mol2":
        return process(
            Chem.MolFromMol2File(
                os.fspath(path),
                sanitize=sanitize,
                removeHs=removeHs,
            )
        )

    if path.suffix == ".sdf":
        return process(
            Chem.SDMolSupplier(
                os.fspath(path),
                sanitize=sanitize,
                removeHs=removeHs,
            )[0]
        )

    if path.suffix == ".pdb":
        return process(
            Chem.MolFromPDBFile(
                os.fspath(path),
                sanitize=sanitize,
                removeHs=removeHs,
            )
        )

    if path.suffix == ".pdbqt":
        with open(path) as file:
            block = "\n".join(line[:66] for line in file)
        return process(
            Chem.MolFromPDBBlock(
                block,
                sanitize=sanitize,
                removeHs=removeHs,
            )
        )

    raise MolFormatError(
        f"supported molecule files are '.mol2', '.sdf', '.pdb' or '.pdbqt', "
        f"but '{path!s}' ends with '{path.suffix!s}'"
    )


def iter_mol_coords(mol: Chem.rdchem.Mol) -> Iterable[tuple[float, float, float]]:
    """Yields atom coordinates of `mol`.

    Args:
        mol: Molecule with atom coordinates.

    Yields:
        Tuple containing atom coordinates.
    """

    conformer = mol.GetConformer()

    for atom in mol.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        yield (pos.x, pos.y, pos.z)


def iter_load_ligands(
    path: Path | str,
    sanitize: bool = False,
    removeHs: bool = True,
) -> Iterable[Chem.rdchem.Mol]:
    """Loads and yields ligands.

    Molecule files containing ligands start with `"ligand"`.

    Args:
        path: Path to directory containing molecule files of ligands.
        sanitize: Whether to sanitize the loaded molecule.
        removeHs: Whether to remove hydrogen atoms from the molecule.

    Yields:
        Loaded ligand as molecule structure.
    """

    path = Path(path)

    processed = {None}

    for p in path.glob("ligand*"):
        if p.stem not in processed:
            try:
                mol = read_mol(p, sanitize, removeHs)
                yield mol
                processed.add(p.stem)
            except MolFormatError as error:
                logger.info(f"Could not read molecule file: {error}")
                raise
