#!/usr/bin/env python3
"""
Fit LJ epsilon/sigma on Mackay LJ13 through ChemFit.fit_anneal.

Eight scaled icosahedra (Ar13, centre plus twelve vertices). The state is
the two parameters, not a 3N geometry. replicas and store are set on the
Fitter call.
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

from chemfit.abstract_objective_function import (
    EvaluateContext,
    QuantityComputerObjectiveFunction,
)
from chemfit.ase_objective_function import SinglePointASEComputer
from chemfit.combined_objective_function import CombinedObjectiveFunction
from chemfit.fitter import Fitter


def loss_function(quants: dict, e_ref: float):
    return (quants["energy"] - e_ref) ** 2


def lj13_term(r: float, eps: float, sigma: float) -> QuantityComputerObjectiveFunction:
    factory = LJ13IcoFactory(r)
    computer = SinglePointASEComputer(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        atoms_factory=factory,
        tag=f"lj13_{r}",
    )
    return QuantityComputerObjectiveFunction(
        loss_function=functools.partial(
            loss_function, e_ref=e_lj_pairs(factory.positions, eps, sigma)
        ),
        quantity_computer=computer,
    )


def main() -> int:
    try:
        import anneal  # noqa: F401, PLC0415
    except ImportError:
        print("anneal is not installed; fit_anneal cannot run", file=sys.stderr)
        return 2

    eps0, sigma0 = 1.0, 1.0
    start = {"epsilon": 2.0, "sigma": 1.5}
    r_min = 2.0 ** (1.0 / 6.0) * sigma0
    r_list = np.linspace(0.95 * r_min, 2.0 * sigma0, 8)
    factory0 = LJ13IcoFactory(float(r_list[0]))
    atoms = factory0()
    n_pairs = len(atoms) * (len(atoms) - 1) // 2
    print("n_atoms", len(atoms), "n_pairs", n_pairs, "n_sizes", len(r_list))
    computer = SinglePointASEComputer(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        atoms_factory=LJ13IcoFactory(float(r_list[0])),
        tag="lj13_preflight",
    )
    quants = computer({"epsilon": eps0, "sigma": sigma0}, EvaluateContext())
    if "energy" not in quants:
        print("preflight missing energy", sorted(quants), file=sys.stderr)
        return 3
    analytic = e_lj_pairs(factory0.positions, eps0, sigma0)
    print("preflight_energy", quants["energy"], "analytic", analytic)
    factories = [LJ13IcoFactory(float(r)) for r in r_list]
    e_refs = [e_lj_pairs(f.positions, eps0, sigma0) for f in factories]
    ob = CombinedObjectiveFunction([lj13_term(float(r), eps0, sigma0) for r in r_list])

    def jac(x: np.ndarray):
        eps, sigma = float(x[0]), float(x[1])
        g = np.zeros(2)
        for factory, e_ref in zip(factories, e_refs):
            energy = e_lj_pairs(factory.positions, eps, sigma)
            d_eps, d_sigma = e_lj_pairs_grad(factory.positions, eps, sigma)
            resid = energy - e_ref
            g[0] += 2.0 * resid * d_eps
            g[1] += 2.0 * resid * d_sigma
        return g

    start_loss = float(ob(start))
    print("start", start, "start_loss", start_loss)
    campaign = Path("lj13-campaign")
    campaign.mkdir(exist_ok=True)
    fitter = Fitter(
        ob,
        initial_params=dict(start),
        bounds={"epsilon": (0.2, 4.0), "sigma": (0.2, 4.0)},
    )
    store = campaign / "params.jsonl"
    if store.is_file():
        store.unlink()
    opt = fitter.fit_anneal(
        budget=240,
        replicas=2,
        store=str(store),
        seed=3,
        history="shared",
        jac=jac,
    )
    end_loss = float(ob(opt))
    n_evals = fitter.contexts[0].n_evals
    print("path hop analytic jac replicas 2")
    print("target", {"epsilon": eps0, "sigma": sigma0})
    print("opt", opt, "end_loss", end_loss, "n_evals", n_evals)
    print("store", store)
    moved = end_loss < start_loss
    recovered = abs(opt["epsilon"] - eps0) < 0.25 and abs(opt["sigma"] - sigma0) < 0.25
    print("LJ13_ANNEAL_OK" if moved and recovered else "LJ13_ANNEAL_MISS")
    return 0 if moved and recovered else 1


if __name__ == "__main__":
    raise SystemExit(main())
