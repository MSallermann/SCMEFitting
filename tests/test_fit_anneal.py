"""fit_anneal on the ChemFit LJ dimer curve and a 4-atom tetrahedron."""

from __future__ import annotations

import functools

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

from conftest import (
    LJAtomsFactory,
    LJTetraFactory,
    apply_params_lj,
    construct_lj,
    e_lj,
    e_lj_tetra,
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


def lj_tetra_term(
    r: float, eps: float, sigma: float
) -> QuantityComputerObjectiveFunction:
    computer = SinglePointASEComputer(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        atoms_factory=LJTetraFactory(r),
        tag=f"lj4_{r}",
    )
    return QuantityComputerObjectiveFunction(
        loss_function=functools.partial(loss_function, e_ref=e_lj_tetra(r, eps, sigma)),
        quantity_computer=computer,
    )


def test_fit_anneal_needs_finite_bounds():
    fitter = Fitter(lambda p: (p["epsilon"] - 1.0) ** 2, {"epsilon": 2.0})
    with pytest.raises(ValueError, match="finite box"):
        fitter.fit_anneal(budget=16, replicas=2)


def test_lj4_tetra_computer_returns_energy():
    r = 2.0 ** (1.0 / 6.0)
    atoms = LJTetraFactory(r)()
    assert len(atoms) == 4
    computer = SinglePointASEComputer(
        calc_factory=construct_lj,
        param_applier=apply_params_lj,
        atoms_factory=LJTetraFactory(r),
        tag="lj4_energy",
    )
    quants = computer({"epsilon": 1.0, "sigma": 1.0}, EvaluateContext())
    assert "energy" in quants
    assert quants["energy"] == pytest.approx(
        e_lj_tetra(r, 1.0, 1.0), rel=1e-6, abs=1e-6
    )


def test_fit_anneal_lj_dimers(tmp_path):
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
        replicas=2,
        store=str(store),
        seed=3,
        history="shared",
    )
    assert store.is_file()
    assert opt["epsilon"] == pytest.approx(eps, rel=0.25, abs=0.25)
    assert opt["sigma"] == pytest.approx(sigma, rel=0.25, abs=0.25)


def test_fit_anneal_lj4_tetrahedron(tmp_path):
    eps, sigma = 1.0, 1.0
    r_min = 2.0 ** (1.0 / 6.0) * sigma
    r_list = np.linspace(0.95 * r_min, 2.0 * sigma, 8)
    ob = CombinedObjectiveFunction([lj_tetra_term(r, eps, sigma) for r in r_list])
    fitter = Fitter(
        ob,
        initial_params={"epsilon": 2.0, "sigma": 1.5},
        bounds={"epsilon": (0.2, 4.0), "sigma": (0.2, 4.0)},
    )
    store = tmp_path / "lj4_tetra.jsonl"
    start_loss = float(ob({"epsilon": 2.0, "sigma": 1.5}))
    opt = fitter.fit_anneal(
        budget=240,
        replicas=2,
        store=str(store),
        seed=3,
        history="shared",
    )
    end_loss = float(ob(opt))
    assert store.is_file()
    assert end_loss < start_loss
    assert opt["epsilon"] == pytest.approx(eps, rel=0.25, abs=0.25)
    assert opt["sigma"] == pytest.approx(sigma, rel=0.25, abs=0.25)
