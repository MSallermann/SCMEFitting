import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones


def e_lj(r: float, eps: float, sigma: float) -> float:
    return 4.0 * eps * ((sigma / r) ** 6 - 1.0) * (sigma / r) ** 6


class LJAtomsFactory:
    def __init__(self, r: float) -> None:
        """Construct two atoms at a distance r."""
        self.p0 = np.zeros(3)
        self.p1 = np.array([r, 0.0, 0.0])

    def __call__(self) -> Atoms:
        return Atoms(positions=[self.p0, self.p1])


class LJTetraFactory:
    """Four Ar atoms on a regular tetrahedron of edge ``r``."""

    def __init__(self, r: float) -> None:
        # Canonical verts have pairwise distance sqrt(8).
        verts = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
        self.positions = verts * (r / np.sqrt(8.0))

    def __call__(self) -> Atoms:
        return Atoms("Ar4", positions=self.positions)


def e_lj_tetra(r: float, eps: float, sigma: float) -> float:
    """Six equal nearest-neighbour pairs, no extra contacts."""
    return 6.0 * e_lj(r, eps, sigma)


def construct_lj(atoms: Atoms):
    atoms.calc = LennardJones(rc=2000)


def apply_params_lj(atoms: Atoms, params: dict[str, float]):
    assert atoms.calc is not None
    assert atoms.calc is not None
    atoms.calc.parameters.sigma = params["sigma"]
    atoms.calc.parameters.epsilon = params["epsilon"]
