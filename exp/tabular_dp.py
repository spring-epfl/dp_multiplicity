import pathlib

import fire
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from diffprivlib.models import (
    RandomForestClassifier as PrivateRandomForestClassifier,
    LogisticRegression as PrivateLogisticRegression,
)

from src.model_set import ModelSet
from uciml import load_dataset


common_model_configs = {
    "lr": dict(penalty=None),
    "rf": dict(n_estimators=100),
}


def train_func(model, X_train, y_train, seed, epsilon=None):
    if model == "lr":
        if epsilon is None:
            clf = LogisticRegression(random_state=seed, **common_model_configs[model])
        else:
            clf = PrivateLogisticRegression(
                epsilon=epsilon, random_state=seed, **common_model_configs[model]
            )

    elif model == "rf":
        if epsilon is None:
            clf = RandomForestClassifier(
                random_state=seed, **common_model_configs[model]
            )
        else:
            clf = PrivateRandomForestClassifier(
                epsilon=epsilon,
                random_state=seed,
                **common_model_configs[model],
            )

    clf.fit(X_train, y_train)
    return clf


def get_data_splits(dataset):
    X, y = load_dataset(dataset, dropna=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, stratify=y, random_state=0
    )
    return X_train, X_test, y_train, y_test


def train(
    dataset: str,
    model: str,
    num_models: int,
    out_path: str = "out",
    epsilon: Optional[float] = None,
    overwrite: bool = False,
    max_num_models: int = 1000,
    condition: str = "beats_baseline",
    n_jobs: int = 4,
):
    X_train, X_test, y_train, y_test = get_data_splits(dataset)
    print(f"{X_train.shape=}, {X_test.shape=}, {y_train.shape=}, {y_test.shape=}")
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"{baseline=}")

    run_name = f"{dataset}_{model}_{epsilon}"
    run_params = {
        "project": "multiplicities",
        "job_type": "train",
        "group": run_name,
        "config": {
            "dataset": dataset,
            "model": model,
            "epsilon": epsilon,
            **common_model_configs[model],
        },
    }
    if condition == "beats_baseline":
        condition_func = lambda clf: clf.score(X_test, y_test) > baseline
    else:
        raise ValueError(f"Unknown condition mode: {condition}")

    model_set = ModelSet(
        model_path=pathlib.Path(out_path) / run_name,
        run_params=run_params,
    )
    model_set.train_until_condition(
        train_func=lambda seed: train_func(
            model, X_train, y_train, seed=seed, epsilon=epsilon
        ),
        condition_func=condition_func,
        eval_func={
            "test_acc": lambda clf: clf.score(X_test, y_test),
            "normalized_adv": lambda clf: 2 * (clf.score(X_test, y_test) - baseline),
            condition: condition_func,
        },
        target_num_models=num_models,
        max_num_models=max_num_models,
        overwrite=overwrite,
        n_jobs=n_jobs,
        verbose=True,
    )


if __name__ == "__main__":
    fire.Fire(train)
