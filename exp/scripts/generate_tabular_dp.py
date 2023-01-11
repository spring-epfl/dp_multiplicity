import itertools


datasets = [
    "contrac",
    "ilpd",
    "audiology",
    "ctg",
    "mammo",
    "dermatology",
    "german-credit",
    "credit-approval",
    "bank",
    "adult",
]
models = ["lr", "rf"]
epsilons = [1.00, 1.25, 1.50, 1.75, 2.00, None]
num_models = 100
n_jobs = 8


for dataset, model, epsilon in itertools.product(datasets, models, epsilons):
    print(
        f"python exp/tabular_dp.py "
        f"--{num_models=} --{n_jobs=} --{dataset=} --{model=} --{epsilon=}"
    )
