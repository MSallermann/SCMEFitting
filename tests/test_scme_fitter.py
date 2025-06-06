from scme_fitting.fitter import Fitter
from scme_fitting.scme_setup import SCMEParams
from scme_fitting.scme_objective_function import (
    EnergyObjectiveFunction,
    DimerDistanceObjectiveFunction,
)
from scme_fitting.utils import dump_dict_to_file
from scme_fitting.multi_energy_objective_function import MultiEnergyObjectiveFunction
from scme_fitting.data_utils import process_csv
import numpy as np
import logging
from pathlib import Path


logging.basicConfig(filename="test_scme_fitter.log", level=logging.DEBUG)

### Common to all tests
PATH_TO_CSV = Path(__file__).parent / "test_configurations_scme/energies.csv"
REFERENCE_CONFIGS, TAGS, REFERENCE_ENERGIES = process_csv(PATH_TO_CSV)

DEFAULT_PARAMS = SCMEParams(
    td=4.780356569608627,
    Ar_OO=299.5695377280358,
    Br_OO=-0.14632711560656822,
    Cr_OO=-2.0071714442805715,
    r_Br=5.867230272424719,
    dms=True,
    qms=True,
)

ADJUSTABLE_PARAMS = ["td", "Ar_OO", "Br_OO", "Cr_OO", "te", "C6", "C8", "C10"]
INITIAL_PARAMS = {k: dict(DEFAULT_PARAMS)[k] for k in ADJUSTABLE_PARAMS}


def test_single_energy_objective_function():
    scme_objective_function = EnergyObjectiveFunction(
        default_scme_params=DEFAULT_PARAMS,
        path_to_scme_expansions=None,
        parametrization_key=None,
        path_to_reference_configuration=REFERENCE_CONFIGS[10],
        reference_energy=REFERENCE_ENERGIES[10],
        tag=TAGS[10],
    )

    fitter = Fitter(
        objective_function=scme_objective_function,
    )

    optimal_params = fitter.fit_scipy(
        initial_parameters=INITIAL_PARAMS, tol=1e-4, options=dict(maxiter=50, disp=True)
    )

    output_folder = Path("test_output_single_energy")
    scme_objective_function.dump_test_configuration(output_folder)

    dump_dict_to_file(output_folder / "optimal_params.json", optimal_params)


def test_dimer_distance_objective_function():
    scme_objective_function = DimerDistanceObjectiveFunction(
        default_scme_params=DEFAULT_PARAMS,
        path_to_scme_expansions=None,
        parametrization_key=None,
        path_to_reference_configuration=REFERENCE_CONFIGS[5],
        dt=1,
        max_steps=500,
        OO_distance_target=3.2,
        tag="dimer_distance",
    )

    fitter = Fitter(
        objective_function=scme_objective_function,
    )

    # optimal_params = fitter.fit_nevergrad(initial_parameters=INITIAL_PARAMS, budget=50)
    optimal_params = fitter.fit_scipy(
        initial_parameters=INITIAL_PARAMS, tol=1e-4, options=dict(maxiter=50, disp=True)
    )

    output_folder = Path("test_output_dimer_distance")
    scme_objective_function.dump_test_configuration(output_folder)

    dump_dict_to_file(output_folder / "optimal_params.json", optimal_params)


def test_multi_energy_ob_function_fitting():
    scme_objective_function = MultiEnergyObjectiveFunction(
        default_scme_params=DEFAULT_PARAMS,
        path_to_scme_expansions=None,
        parametrization_key=None,
        path_to_reference_configuration_list=REFERENCE_CONFIGS,
        reference_energy_list=REFERENCE_ENERGIES,
        tag_list=TAGS,
    )

    fitter = Fitter(
        objective_function=scme_objective_function,
    )

    optimal_params = fitter.fit_scipy(
        initial_parameters=INITIAL_PARAMS, tol=0, options=dict(maxiter=50, disp=True)
    )

    scme_objective_function.write_output(
        "test_output_multi_energy",
        initial_params=INITIAL_PARAMS,
        optimal_params=optimal_params,
    )


if __name__ == "__main__":
    test_single_energy_objective_function()
    test_dimer_distance_objective_function()
    test_multi_energy_ob_function_fitting()
