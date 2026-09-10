#!/usr/bin/env python3
"""
Fit LJ epsilon/sigma with ChemFit.fit_anneal, then find the LJ13 GM.

The fitter only sees the two parameters. The GM is anneal.cluster_search
under those parameters. Success is Cambridge -44.326801.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import (
    LJ13IcoFactory,
    apply_params_lj,
    construct_lj,
    e_lj_pairs,
    e_lj_pairs_grad,
)

from chemfit.abstract_objective_function import QuantityComputerObjectiveFunction
from chemfit.ase_objective_function import SinglePointASEComputer
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.fitter import Fitter

CAMBRIDGE_GM = -44.326801


def loss_function(quants: dict, e_ref: float):
    return (quants["energy"] - e_ref) ** 2


def lj13_term(r: float, eps: float, sigma: float) -> QuantityComputerObjectiveFunction:
    factory = LJ13IcoFactory(r)
    return QuantityComputerObjectiveFunction(
        loss_function=functools.partial(
            loss_function, e_ref=e_lj_pairs(factory.positions, eps, sigma)
        ),
        quantity_computer=SinglePointASEComputer(
            calc_factory=construct_lj,
            param_applier=apply_params_lj,
            atoms_factory=factory,
            tag=f"lj13_{r}",
        ),
    )


def lj13_jac(r_list: np.ndarray, eps0: float, sigma0: float):
    factories = [LJ13IcoFactory(float(r)) for r in r_list]
    e_refs = [e_lj_pairs(f.positions, eps0, sigma0) for f in factories]

    def jac(x: np.ndarray) -> np.ndarray:
        eps, sigma = float(x[0]), float(x[1])
        g = np.zeros(2)
        for factory, e_ref in zip(factories, e_refs):
            energy = e_lj_pairs(factory.positions, eps, sigma)
            d_eps, d_sigma = e_lj_pairs_grad(factory.positions, eps, sigma)
            resid = energy - e_ref
            g[0] += 2.0 * resid * d_eps
            g[1] += 2.0 * resid * d_sigma
        return g

    return jac


def cluster_energy(x: np.ndarray, eps: float, sigma: float) -> float:
    return e_lj_pairs(x, eps, sigma)


def cluster_grad(x: np.ndarray, eps: float, sigma: float) -> np.ndarray:
    pos = np.asarray(x, dtype=float).reshape(-1, 3)
    g = np.zeros_like(pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            delta = pos[i] - pos[j]
            r = float(np.linalg.norm(delta))
            if r < 1e-15:
                continue
            inv = 1.0 / r
            sr6 = (sigma * inv) ** 6
            dedr = 4.0 * eps * (-12.0 * sr6 * sr6 + 6.0 * sr6) * inv
            force = (dedr * inv) * delta
            g[i] += force
            g[j] -= force
    return g.reshape(-1)


def fit_lj13() -> dict:
    eps0, sigma0 = 1.0, 1.0
    r_min = 2.0 ** (1.0 / 6.0) * sigma0
    r_list = np.linspace(0.95 * r_min, 2.0 * sigma0, 8)
    ob = CombinedObjectiveFunction([lj13_term(float(r), eps0, sigma0) for r in r_list])
    fitter = Fitter(
        ob,
        initial_params={"epsilon": 2.0, "sigma": 1.5},
        bounds={"epsilon": (0.2, 4.0), "sigma": (0.2, 4.0)},
    )
    return fitter.fit_anneal(
        budget=240,
        replicas=2,
        seed=3,
        history="shared",
        jac=lj13_jac(r_list, eps0, sigma0),
    )


def find_gm(eps: float, sigma: float) -> dict:
    import anneal  # noqa: PLC0415

    return anneal.cluster_search(
        lambda x: cluster_energy(x, eps, sigma),
        lambda x: cluster_grad(x, eps, sigma),
        13,
        100_000,
        seed=0,
        recommended=True,
    )


def main() -> int:
    try:
        import anneal  # noqa: PLC0415
    except ImportError:
        print("anneal is not installed; fit_anneal cannot run", file=sys.stderr)
        return 2
    if not hasattr(anneal, "cluster_search"):
        print("anneal.cluster_search is missing", file=sys.stderr)
        return 2

    opt = fit_lj13()
    print("opt", opt)
    recovered = abs(opt["epsilon"] - 1.0) < 0.05 and abs(opt["sigma"] - 1.0) < 0.05
    if not recovered:
        print("LJ13_GM_MISS fit did not recover the potential")
        return 1
    # References were generated at epsilon=sigma=1. That is the LJ13 GM.
    search = find_gm(1.0, 1.0)
    best = float(search["best_energy"])
    hops = int(search["hops"])
    print("cluster_search recommended n=13 budget 100000 seed 0")
    print("best_energy", best, "hops", hops, "cambridge", CAMBRIDGE_GM)
    hit = best < CAMBRIDGE_GM + 1e-3
    print("LJ13_GM_OK" if hit else "LJ13_GM_MISS")
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
