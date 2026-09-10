"""fit_anneal on the ChemFit LJ dimer curve and a Mackay LJ13 icosahedron."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import (
    LJ13IcoFactory,
    LJAtomsFactory,
    apply_params_lj,
    construct_lj,
    e_lj,
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

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("anneal")


def loss_function(quants: dict, e_ref: float):
    return (quants["energy"] - e_ref) ** 2


def lj_ob_term(r: float, eps: float, sigma: float) -> QuantityComputerObjectiveFunction:
    computer = SinglePointASEComputer(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        atoms_factory=LJAtomsFactory(r),
        tag=f"lj_{r}",
    )
    return QuantityComputerObjectiveFunction(
        loss_function=functools.partial(loss_function, e_ref=e_lj(r, eps, sigma)),
        quantity_computer=computer,
    )


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


def test_fit_anneal_needs_finite_bounds():
    fitter = Fitter(lambda p: (p["epsilon"] - 1.0) ** 2, {"epsilon": 2.0})
    with pytest.raises(ValueError, match="finite box"):
        fitter.fit_anneal(budget=16, replicas=2)


def test_lj13_computer_returns_energy():
    r = 2.0 ** (1.0 / 6.0)
    atoms = LJ13IcoFactory(r)()
    assert len(atoms) == 13
    computer = SinglePointASEComputer(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        atoms_factory=LJ13IcoFactory(r),
        tag="lj13_energy",
    )
    quants = computer({"epsilon": 1.0, "sigma": 1.0}, EvaluateContext())
    assert "energy" in quants
    assert quants["n_atoms"] == 13
    analytic = e_lj_pairs(LJ13IcoFactory(r).positions, 1.0, 1.0)
    assert quants["energy"] == pytest.approx(analytic, rel=1e-6, abs=1e-6)


def test_fit_anneal_lj_dimers(tmp_path: Path):
    eps, sigma = 1.0, 1.0
    r_min = 2.0 ** (1.0 / 6.0) * sigma
    r_list = np.linspace(0.95 * r_min, 2.0 * sigma, 8)
    ob = CombinedObjectiveFunction([lj_ob_term(r, eps, sigma) for r in r_list])
    fitter = Fitter(
        ob,
        initial_params={"epsilon": 2.0, "sigma": 1.5},
        bounds={"epsilon": (0.2, 4.0), "sigma": (0.2, 4.0)},
    )
    store = tmp_path / "lj_dimers.jsonl"
    opt = fitter.fit_anneal(
        budget=240,
        replicas=1,
        store=str(store),
        seed=3,
        history="shared",
        jac=False,
    )
    assert store.is_file()
    assert opt["epsilon"] == pytest.approx(eps, rel=0.25, abs=0.25)
    assert opt["sigma"] == pytest.approx(sigma, rel=0.25, abs=0.25)


def test_fit_anneal_lj13_icosahedron(tmp_path: Path):
    eps, sigma = 1.0, 1.0
    r_min = 2.0 ** (1.0 / 6.0) * sigma
    r_list = np.linspace(0.95 * r_min, 2.0 * sigma, 8)
    factories = [LJ13IcoFactory(float(r)) for r in r_list]
    e_refs = [e_lj_pairs(f.positions, eps, sigma) for f in factories]
    ob = CombinedObjectiveFunction([lj13_term(float(r), eps, sigma) for r in r_list])

    def jac(x: np.ndarray):
        e, s = float(x[0]), float(x[1])
        g = np.zeros(2)
        for factory, e_ref in zip(factories, e_refs):
            energy = e_lj_pairs(factory.positions, e, s)
            d_eps, d_sigma = e_lj_pairs_grad(factory.positions, e, s)
            resid = energy - e_ref
            g[0] += 2.0 * resid * d_eps
            g[1] += 2.0 * resid * d_sigma
        return g

    fitter = Fitter(
        ob,
        initial_params={"epsilon": 2.0, "sigma": 1.5},
        bounds={"epsilon": (0.2, 4.0), "sigma": (0.2, 4.0)},
    )
    store = tmp_path / "lj13_ico.jsonl"
    start_loss = float(ob({"epsilon": 2.0, "sigma": 1.5}))
    opt = fitter.fit_anneal(
        budget=240,
        replicas=2,
        store=str(store),
        seed=3,
        history="shared",
        jac=jac,
    )
    end_loss = float(ob(opt))
    assert store.is_file()
    assert end_loss < start_loss
    assert opt["epsilon"] == pytest.approx(eps, rel=0.05, abs=0.05)
    assert opt["sigma"] == pytest.approx(sigma, rel=0.05, abs=0.05)
