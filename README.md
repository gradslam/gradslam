# gradslam

Development repo for the `gradslam` library.

[![CircleCI](https://circleci.com/gh/gradslam/gradslam.svg?style=shield&circle-token=109c43f395121b987111c85a9cf51d5fd75ea72c)](https://circleci.com/gh/gradslam/gradslam/tree/master)

[![CircleCI](https://circleci.com/gh/gradslam/gradslam.svg?style=shield&circle-token=109c43f395121b987111c85a9cf51d5fd75ea72c)](https://circleci.com/gh/gradslam/gradslam/tree/master)



## Building the package

In a `conda` environment (or a `virtualenv` environment if you prefer), install PyTorch (version `1.3.0` or greater). Then, `gradslam` can be installed by navigating into the base directory of this repo (i.e., the directory containing this readme file) and executing the following command.

```bash
python setup.py build develop
```

## Verifying the installation

To verify if `gradslam` has successfully been built, fire up the python interpreter, and import!

```py
import gradslam as gs
print(gs.__version__)
```

You should see the version number displayed.

## Running the unittests

From the base directory of the repo, run the following command.

```bash
pytest tests/
```

### Get coverage info

To get stats (in particular test coverage ratio), run

```bash
pytest test/ --cov
```

## Build docs

To build sphinx documentation, execute the following commands (**AFTER** building the `gradslam` package).

```bash
cd docs
sphinx-build . _build
```

This should build the docs in `docs/_build`. Open `docs/_build/index.html` in your web browser to access the docs.
