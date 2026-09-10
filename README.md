<p align="center">
    <img src="https://github.com/msallermann/chemfit/blob/next/logo/chemfit_logo_portable.svg?raw=true" width="400"/>
</p>

# About

ChemFit is a Python package for concurrent force-field parameter optimization. It can be used with ASE calculators and external executables.

# Documentation

Please check the **documentation** for details [here](https://chemfit.readthedocs.io).


# Installation

From PyPi:

```bash
pip install chemfit
```

Or, locally:

```bash
git clone git@github.com:MSallermann/chemfit.git
pip install chemfit
```

# Anneal backend

`Fitter.fit_anneal` is the same flatten as `fit_scipy`, then
[`anneal.minimize`](https://github.com/HaoZeke/anneal). Set `replicas`
and `store` on that call. `examples/lj4_anneal.py` fits LJ
`epsilon`/`sigma` on eight ASE 4-atom tetrahedra.

```bash
pip install chemfit anneal
python examples/lj4_anneal.py
```

# Citation

If you find ChemFit useful and happen to use it in any academic context, please use this reference to cite it:

```
@misc{sallermann2026chemfitframeworkautomatedhighdimensional,
      title={ChemFit: A framework for automated high-dimensional model parameter optimization},
      author={Moritz Sallermann and Amrita Goswami and Rosana Collepardo-Guevara and Alberto Ocana and Hannes Jónsson and Elvar Ö. Jónsson and Jorge R. Espinosa},
      year={2026},
      eprint={2603.11769},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2603.11769},
}
```

Thanks!

# Problems?

Please open an issue [here](https://github.com/MSallermann/chemfit/issues).
