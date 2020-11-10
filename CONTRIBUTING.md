# Contributing to gradslam
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

However, if you're adding any significant features, please make sure to have a corresponding issue to outline your proposal and motivation and allow time for us to give feedback, *before* you send a PR.
We do not always accept new features, and we take the following factors into consideration:

- Whether the same feature can be achieved without modifying gradslam directly. If any aspect of the API is not extensible, please highlight this in an issue so we can work on making this more extensible.
- Whether the feature is potentially useful to a large audience, or only to a small portion of users.
- Whether the proposed solution has a good design and interface.
- Whether the proposed solution adds extra mental/practical overhead to users who don't need such feature.
- Whether the proposed solution breaks existing APIs.

When sending a PR, please ensure you complete the following steps:

1. Fork the repo and create your branch from `main`. Install gradslam with developer dependencies and in editable mode:
```
git clone https://github.com/gradslam/gradslam.git
cd gradslam
git checkout -b feat/foo_feature  # or git checkout -b fix/bar_bug
pip install -e .[dev]
```
2. If you've added code that should be tested, add tests in `tests/` directory.
3. If you've changed any APIs, please update the documentation.
4. Ensure the test suite passes by running `pytest tests/` from the project root.
5. Ensure the documentation builds by running `cd docs/ && sphinx-build . _build` from the project root. This should build the documentation in `docs/_build`. Open `docs/_build/index.html` in your web browser to access the documentation.
5. Make sure your code lints by running `black gradslam/ && flake8 gradslam/` from the project root.
6. If a PR contains multiple orthogonal changes, split it into multiple separate PRs.

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Coding Style  
We follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008) and [type function input and output](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#functions).

For the linter to work, you will need to install `black`, `flake8` and `isort`, and they need to be fairly up to date.

## License
By contributing to gradslam, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.