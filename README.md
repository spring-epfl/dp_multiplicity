

## Installation

```
pipenv install
```

## Launching Experiments

The experiments rely on Weights & Biases (wandb) for experiment tracking.

Launching goes in two steps. First, generate the bash scripts.
```
argmapper sweep exp/config/tabular_dp_experiments.yml --out exp/config/tabular_dp_experiments.runfile
argmapper sweep exp/config/cifar10_dp_experiments.yml --out exp/config/tabular_dp_experiments.runfile
```

Next, launch the commands in the generated bash scripts in any way you prefer. You can use:
```
argmapper launch exp/config/tabular_dp_experiments.runfile
argmapper launch exp/config/cifar10_dp_experiments.runfile
```

Keep in mind you need to have wandb set up and running so that the experiment metadata are saved.


## Analyzing Experiments

Analyses of the experiments are done using Jupyter notebooks in the `notebooks` directory.
They are saved as Python scripts in git. Use `jupytext <notebook>.py --to ipynb` to regenerate the
notebook `*.ipynb`. Other than the analysis of experiments, the notebooks directory contains
standalone experiments with synthetic data and theoretical expressions.
