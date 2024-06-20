import logging
import tempfile
import subprocess
import numpy as np

from pathlib import Path
from rdkit import Chem
from numpy.typing import NDArray
from Bio.PDB import Selection
from Bio.PDB.ResidueDepth import _get_atom_radius, _read_vertex_array

logger = logging.getLogger(__name__)


def get_surface(
    model: Chem.rdchem.Mol,
    MSMS: str = "msms",
) -> NDArray[np.float64]:
    """Represent molecular surface as a vertex list array.

    Reimplementation of :func:`Bio.PDB.ResidueDepth.get_surface` in order to
    redirect stderr to warnings.

    Args:
        model: BioPython PDB model object (used to get atoms for input model).
        MSMS: MSMS executable (used as argument to subprocess.run).
        verbose: Whether to output errors.

    Returns:
        NumPy array that represents the vertex list of the molecular surface.

    Raises:
        RuntimeError: If generate surface file command failed.
    """

    atom_list = Selection.unfold_entities(model, "A")

    with tempfile.TemporaryDirectory() as tempdir:
        temp_dir = Path(tempdir)
        xyz_path = temp_dir / "xyz"

        with open(xyz_path, "w") as pdb_to_xyzr:
            for atom in atom_list:
                x, y, z = atom.coord
                radius = _get_atom_radius(atom, rtype="united")
                pdb_to_xyzr.write(f"{x:6.3f}\t{y:6.3f}\t{z:6.3f}\t{radius:1.2f}\n")

        surface_path = temp_dir / "surface"
        cmd = [MSMS, "-probe_radius", "1.5", "-if", xyz_path, "-of", surface_path]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        with process.stderr:
            for line in iter(process.stderr.readline, b""):
                logger.warning(f"{MSMS}: {line.decode()}")
        error = process.wait()

        surface_vert_path = surface_path.parent / (surface_path.name + ".vert")

        if error or not surface_vert_path.exists():
            raise RuntimeError(
                f"failed to generate surface file with command '{' '.join(map(str, cmd))}'"
            )

        surface = _read_vertex_array(surface_vert_path)

    return surface
