# neigh-ai

[![Build status](https://img.shields.io/github/actions/workflow/status/hrsather/neigh-ai/main.yml?branch=main)](https://github.com/hrsather/neigh-ai/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/hrsather/neigh-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/hrsather/neigh-ai)
[![Commit activity](https://img.shields.io/github/commit-activity/m/hrsather/neigh-ai)](https://img.shields.io/github/commit-activity/m/hrsather/neigh-ai)

This repo helps buy winning horses.

Install the environment and the pre-commit hooks with

```bash
make install
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To verify your code follows our formatting and types rules:

```bash
make check
```

To verify your tests pass:

```bash
make test
```

To run locally:

```bash
docker build -t dashboard .
docker run  -p 8050:8050  -v $(pwd)/data:/code/data dashboard
```

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).
