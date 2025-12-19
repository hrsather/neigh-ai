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

Mean baseline: 0.380

race_score_sire
MSE - Model: 0.214

avg_sire_sibling_score
MSE - Model: 0.292

max_dam_sibling_score
MSE - Model: 0.103

max_sire_sibling_score
MSE - Model: 0.292

std_dam_sibling_score
MSE - Model: 0.094

std_sire_sibling_score
MSE - Model: 0.356

avg_siresire_cousin_score
MSE - Model: 0.213

avg_siredam_cousin_score
MSE - Model: 0.211

max_siresire_cousin_score
MSE - Model: 0.213

max_siredam_cousin_score
MSE - Model: 0.211

std_siresire_cousin_score
MSE - Model: 0.212

std_siredam_cousin_score
MSE - Model: 0.208

min_siresire_cousin_score
MSE - Model: 0.213

min_siredam_cousin_score
MSE - Model: 0.211

avg_siresire_auntuncle_score
MSE - Model: 0.214

avg_siredam_auntuncle_score
MSE - Model: 0.214

max_siresire_auntuncle_score
MSE - Model: 0.214

max_siredam_auntuncle_score
MSE - Model: 0.214

min_siresire_auntuncle_score
MSE - Model: 0.214

min_siredam_auntuncle_score
MSE - Model: 0.214

std_siresire_auntuncle_score
MSE - Model: 0.213

std_siredam_auntuncle_score
MSE - Model: 0.211

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).
