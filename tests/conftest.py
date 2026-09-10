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


class LJ13IcoFactory:
    """Mackay LJ13: centre plus twelve vertices, centre-to-vertex ``r``.

    Same construction as anneal's ``icosahedron13`` (golden-ratio verts,
    scaled so the first shell sits at ``r``). All 78 pairs contribute.
    """

    def __init__(self, r: float) -> None:
        phi = (1.0 + 5.0**0.5) / 2.0
        verts = np.array(
            [
                [0.0, 1.0, phi],
                [0.0, 1.0, -phi],
                [0.0, -1.0, phi],
                [0.0, -1.0, -phi],
                [1.0, phi, 0.0],
                [1.0, -phi, 0.0],
                [-1.0, phi, 0.0],
                [-1.0, -phi, 0.0],
                [phi, 0.0, 1.0],
                [-phi, 0.0, 1.0],
                [phi, 0.0, -1.0],
                [-phi, 0.0, -1.0],
            ]
        )
        scale = r / np.sqrt(1.0 + phi * phi)
        self.positions = np.vstack([np.zeros(3), verts * scale])

    def __call__(self) -> Atoms:
        return Atoms("Ar13", positions=self.positions)


def e_lj_pairs(positions: np.ndarray, eps: float, sigma: float) -> float:
    """All-pair Lennard-Jones energy of a point set."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    energy = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            energy += e_lj(float(np.linalg.norm(pos[i] - pos[j])), eps, sigma)
    return energy


def construct_lj(atoms: Atoms):
    atoms.calc = LennardJones(rc=2000)


def apply_params_lj(atoms: Atoms, params: dict[str, float]):
    assert atoms.calc is not None
    assert atoms.calc is not None
    atoms.calc.parameters.sigma = params["sigma"]
    atoms.calc.parameters.epsilon = params["epsilon"]
