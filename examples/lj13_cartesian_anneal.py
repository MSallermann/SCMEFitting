#!/usr/bin/env python3
"""
ChemFit.fit_anneal on LJ13 with the 39 Cartesian coordinates as parameters.

The loss is the pair energy at epsilon = sigma = 1. This is the
high-dimensional ChemFit call, not a two-parameter fit plus cluster_search.
The gate is Cambridge -44.326801.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import LJ13IcoFactory, e_lj

from chemfit.fitter import Fitter

CAMBRIDGE_GM = -44.326801
N_ATOMS = 13
DIM = 3 * N_ATOMS
KEYS = [f"c{i:02d}" for i in range(DIM)]
BOX = (-8.0, 8.0)
EPS = 1.0
SIGMA = 1.0
R_MIN = 1e-8


def vec_from_params(params: dict) -> np.ndarray:
    return np.array([float(params[k]) for k in KEYS], dtype=float)


def params_from_vec(x: np.ndarray) -> dict[str, float]:
    flat = np.asarray(x, dtype=float).reshape(-1)
    if flat.size != DIM:
        msg = f"expected {DIM} coordinates, got {flat.size}"
        raise ValueError(msg)
    return {k: float(v) for k, v in zip(KEYS, flat)}


def pair_energy(x: np.ndarray) -> float:
    pos = np.asarray(x, dtype=float).reshape(N_ATOMS, 3)
    energy = 0.0
    for i in range(N_ATOMS):
        for j in range(i + 1, N_ATOMS):
            r = max(float(np.linalg.norm(pos[i] - pos[j])), R_MIN)
            energy += e_lj(r, EPS, SIGMA)
    return float(energy)


class _AskDump:
    fh = None


def write_min_line(path: Path, energy: float, coords: np.ndarray) -> None:
    flat = np.asarray(coords, dtype=float).reshape(-1)
    path.write_text(f"{energy:.8f} " + " ".join(f"{v:.6f}" for v in flat) + "\n")


def loss(params: dict) -> float:
    x = vec_from_params(params)
    energy = pair_energy(x)
    if _AskDump.fh is not None:
        _AskDump.fh.write(f"{energy:.8f} " + " ".join(f"{v:.6f}" for v in x) + "\n")
    return energy


def jac(x: np.ndarray) -> np.ndarray:
    pos = np.asarray(x, dtype=float).reshape(N_ATOMS, 3)
    g = np.zeros_like(pos)
    for i in range(N_ATOMS):
        for j in range(i + 1, N_ATOMS):
            delta = pos[i] - pos[j]
            r = max(float(np.linalg.norm(delta)), R_MIN)
            inv = 1.0 / r
            sr6 = (SIGMA * inv) ** 6
            dedr = 4.0 * EPS * (-12.0 * sr6 * sr6 + 6.0 * sr6) * inv
            force = (dedr * inv) * delta
            g[i] += force
            g[j] -= force
    return g.reshape(-1)


def random_start(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.empty((N_ATOMS, 3), dtype=float)
    for i in range(N_ATOMS):
        for _ in range(1000):
            cand = rng.uniform(-2.5, 2.5, size=3)
            if i == 0 or np.min(np.linalg.norm(pos[:i] - cand, axis=1)) >= 0.75:
                pos[i] = cand
                break
        else:
            pos[i] = rng.uniform(-2.5, 2.5, size=3)
    return pos.reshape(-1)


def main() -> int:
    try:
        import anneal  # noqa: F401
    except ImportError:
        print("anneal is not installed; fit_anneal cannot run", file=sys.stderr)
        return 2

    ico = LJ13IcoFactory(2.0 ** (1.0 / 6.0) * SIGMA).positions.reshape(-1)
    ico_e = pair_energy(ico)
    random_x = random_start(1)
    g = jac(ico)
    e0 = pair_energy(ico)
    step = 1e-6
    fd0 = (pair_energy(ico + np.array([step] + [0.0] * (DIM - 1))) - e0) / step
    print("ico_energy", ico_e)
    print("random_energy", pair_energy(random_x))
    print("n_params", DIM)
    print("jac0", float(g[0]), "fd0", float(fd0))

    campaign = Path("lj13-cartesian").resolve()
    campaign.mkdir(exist_ok=True)
    rows = []
    hit_any = False
    gm_x = None
    gm_e = None
    for name, x0, budget in (
        ("ico", ico, 20_000),
        ("random", random_x, 20_000),
    ):
        start_e = pair_energy(x0)
        dump_path = campaign / f"{name}.min"
        print("start", name, start_e, "dump", dump_path)
        with dump_path.open("w", encoding="ascii") as dump_fh:
            _AskDump.fh = dump_fh
            fitter = Fitter(
                loss,
                initial_params=params_from_vec(x0),
                bounds=dict.fromkeys(KEYS, BOX),
            )
            opt = fitter.fit_anneal(
                budget=budget,
                replicas=2,
                seed=3,
                history="shared",
                jac=jac,
            )
            _AskDump.fh = None
        best_x = vec_from_params(opt)
        best = pair_energy(best_x)
        n_evals = int(fitter.contexts[0].n_evals)
        ctx_best = fitter.contexts[0].opt_loss
        hit = best < CAMBRIDGE_GM + 1e-3
        hit_any = hit_any or hit
        if hit and gm_x is None:
            gm_x = best_x
            gm_e = best
        dump_lines = sum(
            1 for line in dump_path.read_text().splitlines() if line.strip()
        )
        print(
            "result",
            name,
            "best_energy",
            best,
            "cambridge",
            CAMBRIDGE_GM,
            "n_evals",
            n_evals,
            "ctx_best",
            ctx_best,
            "dump_lines",
            dump_lines,
        )
        rows.append(
            {
                "start": name,
                "start_energy": start_e,
                "best_energy": best,
                "n_evals": n_evals,
                "ctx_best": ctx_best,
                "dump_lines": dump_lines,
                "hit": hit,
            }
        )
    if gm_x is not None and gm_e is not None:
        write_min_line(campaign / "ref.min", gm_e, gm_x)
    (campaign / "opt.json").write_text(
        json.dumps(
            {
                "n_params": DIM,
                "ico_energy": ico_e,
                "cambridge": CAMBRIDGE_GM,
                "runs": rows,
                "hit": hit_any,
            }
        )
        + "\n"
    )
    print("LJ13_CART_OK" if hit_any else "LJ13_CART_MISS")
    return 0 if hit_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
