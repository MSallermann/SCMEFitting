#!/usr/bin/env python3
"""
Fit LJ epsilon/sigma on Mackay LJ13 with Fitter.fit_anneal.

Same ChemFit object as tests/test_lj.py: CombinedObjectiveFunction of
ASE single-points. The optimizer is anneal instead of SciPy.
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


def main() -> int:
    try:
        import anneal  # noqa: PLC0415
    except ImportError:
        print("anneal is not installed; fit_anneal cannot run", file=sys.stderr)
        return 2
    del anneal

    eps0, sigma0 = 1.0, 1.0
    start = {"epsilon": 2.0, "sigma": 1.5}
    r_min = 2.0 ** (1.0 / 6.0) * sigma0
    r_list = np.linspace(0.95 * r_min, 2.0 * sigma0, 8)
    ob = CombinedObjectiveFunction([lj13_term(float(r), eps0, sigma0) for r in r_list])
    fitter = Fitter(
        ob,
        initial_params=dict(start),
        bounds={"epsilon": (0.2, 4.0), "sigma": (0.2, 4.0)},
    )
    opt = fitter.fit_anneal(
        budget=240,
        replicas=2,
        seed=3,
        history="shared",
        jac=lj13_jac(r_list, eps0, sigma0),
    )
    print("start", start)
    print("target", {"epsilon": eps0, "sigma": sigma0})
    print("opt", opt)
    ok = abs(opt["epsilon"] - eps0) < 1e-6 and abs(opt["sigma"] - sigma0) < 1e-6
    print("LJ13_ANNEAL_OK" if ok else "LJ13_ANNEAL_MISS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
