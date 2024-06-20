import os
import subprocess
from typing import Iterable
from pathlib import Path
from contextlib import contextmanager
from Bio.PDB import PDBParser, PDBIO, Select

from ..constants import ALLOWABLE_RESIDUES, ALLOWABLE_ELEMENTS


def mol2_to_pdb(src: Path, dst: Path, verbose=False) -> None:
    """ """

    if verbose:
        stderr = subprocess.STDERR
    else:
        stderr = subprocess.DEVNULL

    subprocess.run(
        ["babel", "-imol2", os.fspath(src), "-opdb", os.fspath(dst)],
        stderr=stderr,
    )


class _SelectResiduesAndElements(Select):
    """ """

    def __init__(
        self,
        allowable_residues=ALLOWABLE_RESIDUES,
        allowable_elements=ALLOWABLE_ELEMENTS,
    ) -> None:
        """ """

        self.allowable_residues = allowable_residues
        self.allowable_elements = allowable_elements

    def accept_residue(self, residue):
        return residue.resname in self.allowable_residues

    def accept_atom(self, atom):
        return atom.element in self.allowable_elements


def filter_pdb(
    src: Path,
    dst: Path,
    allowable_residues=ALLOWABLE_RESIDUES,
    allowable_elements=ALLOWABLE_ELEMENTS,
) -> None:
    """ """

    name = src.parent.name

    parser = PDBParser()
    struct = parser.get_structure(name, os.fspath(src))

    io = PDBIO()
    io.set_structure(struct)
    io.save(
        os.fspath(dst),
        _SelectResiduesAndElements(allowable_residues, allowable_elements),
    )


@contextmanager
def prepare_pdb(
    path: str,
    allowable_residues: list[str] = ALLOWABLE_RESIDUES,
    allowable_elements: list[str] = ALLOWABLE_ELEMENTS,
) -> Iterable[Path]:
    """ """

    path = Path(path)

    p_mol2 = path / "protein.mol2"
    p_pdb = path / "protein.pdb"
    p_pdb_cleaned = path / "protein.cleaned.pdb"

    pdb_was_generated = False

    if p_mol2.exists() and not p_pdb.exists():
        mol2_to_pdb(p_mol2, p_pdb)
        pdb_was_generated = True

    filter_pdb(
        p_pdb,
        p_pdb_cleaned,
        allowable_residues,
        allowable_elements,
    )

    try:
        yield p_pdb_cleaned
    finally:
        if pdb_was_generated:
            try:
                os.remove(p_pdb)
            except FileNotFoundError:
                pass
        try:
            os.remove(p_pdb_cleaned)
        except FileNotFoundError:
            pass
