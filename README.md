<!--
<p align="center">
  <img src="https://github.com/vgvinter/biomassml/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  biomassml
</h1>

<p align="center">
    <a href="https://github.com/vgvinter/biomassml/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/vgvinter/biomassml/workflows/Tests/badge.svg" />
    </a>
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-python--package-yellow" /> 
    </a>
    <a href="https://pypi.org/project/biomassml">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/biomassml" />
    </a>
    <a href="https://pypi.org/project/biomassml">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/biomassml" />
    </a>
    <a href="https://github.com/vgvinter/biomassml/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/biomassml" />
    </a>
    <a href='https://biomassml.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/biomassml/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://zenodo.org/badge/latestdoi/455565709"><img src="https://zenodo.org/badge/455565709.svg" alt="DOI"></a>
</p>

Predicting gasification results for biomasses.

## 💪 Getting Started

biomassml allows you to predict gasification results for biomass samples as a function of the biomass properties and the process operating conditions.

> TODO show in a very small amount of space the **MOST** useful thing your package can do.
Make it as short as possible! You have an entire set of docs for later.

### Command Line Interface

The biomassml command line tool is automatically installed. It can
be used from the shell with the `--help` flag to show all subcommands:

```shell
$ biomassml --help
```

> TODO show the most useful thing the CLI does! The CLI will have documentation auto-generated
by `sphinx`.

## 🚀 Installation

<!-- Uncomment this section after your first ``tox -e finish``
The most recent release can be installed from
[PyPI](https://pypi.org/project/biomassml/) with:

```bash
$ pip install biomassml
```
-->

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/vgvinter/biomassml.git
```

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/vgvinter/biomassml.git
$ cd biomassml
$ pip install -e .
```

## :memo: Description

### Datasets

The data for training the models can be found in `data/data_GASIF_biomass.csv`, which contains data on biomass properties, gasification operating conditions, and results of the gasification process.

The data for the new biomasses used in this work to predict gasification results can be found in `data/data_NEW_biomasses.csv`, which contains data on the properties of different biomasses collected from the literature.

### Training

The code to build the Gaussian Processes Regression (GPR) models used in this work can be found in `src/biomassml/build_model.py`. Code to build single-output GPR and coregionalized GPR models is included. The default configuration for training can be found in `src/biomassml/conf/default.yaml`.

Helper functions to perform leave-one-out cross-validation (LOOCV) on the model and to calculate metrics can be found in `src/biomassml/metrics.py`. Functions to perform LOOCV on a given kernel can be found in `src/biomassml/pipeline.py`.

### Feature importance

The command-line-tools for the analysis of the feature importance can be found in `src/biomassml/feature_importance.py`, including partial dependency plots and SHAP analysis.`

### Predictions

The code to predict outputs can be found in `src/biomassml/predict_outputs.py`. Functions to predict the biomass gasification outputs used in this work can be found in `src/biomassml/predict_outputs.py`.

### Trained models

We provide the Gaussian Processes Regression (GPR) models trained in this work in the `models` directory. The model trained using leave-one-out cross-validation (LOOCV) can be found in `models/model_GPR_loocv`. The model retrained on all data can be found in `models/model_GPR_retrained`.

### Example usage

The use of the main functions of this package is shown in Jupyter Notebooks in the `notebooks` directory. The training of the Gaussian Processes Regression (GPR) models used in this work can be found in `notebooks/train_GPR_model.ipynb`. The analysis of the feature importance can be found in `notebooks/feature_importance.ipynb`. The prediction of the gasification results for different biomasses from the literature can be found in `notebooks/predictions_new_dataset.ipynb`. The cluster analysis can be found in `notebooks/cluster_analysis.ipynb`.

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com/vgvinter/biomassml/blob/main/CONTRIBUTING.rst) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

<!--
### 📖 Citation

Citation goes here!
-->

<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

-->


### 💰 Funding

This work was carried out with financial support from the Spanish Agencia Estatal de Investigación (AEI) through Grant TED2021-131693B-I00 funded by MCIN/AEI/ 10.13039/501100011033 and by the “European Union NextGenerationEU/PRTR”, and from the Spanish National Research Council (CSIC) through Programme for internationalization i-LINK 2021 (Project LINKA20412).

<!--
This project has been supported by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
-->

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instrutions</summary>

  
The final section of the README is for if you want to get involved by making a code contribution.

### ❓ Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/kjappelbaum/biomassml/actions?query=workflow%3ATests).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses BumpVersion to switch the version number in the `setup.cfg` and
   `src/biomassml/version.py` to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel
3. Uploads to PyPI using `twine`. Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.
</details>
