import pathlib
import pytest
import tempfile
import warnings
import numpy as np

from model_set import ModelSet, PickleSerializer, DillSerializer


@pytest.fixture
def model_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture(params=["pickle", "dill"])
def serializer(request):
    if request.param == "pickle":
        return PickleSerializer()
    elif request.param == "dill":
        return DillSerializer()


@pytest.mark.parametrize("verbose", [False, True])
def test_train(model_dir, serializer, verbose):
    model_set = ModelSet(model_path=model_dir, serializer=serializer)
    assert not (model_dir / model_set.model_filename_template.format(seed=49)).exists()
    train_func = lambda seed: "whatever"
    model_set.train(train_func, seeds=50, verbose=verbose)
    assert (model_dir / model_set.model_filename_template.format(seed=49)).exists()
    assert not (model_dir / model_set.model_filename_template.format(seed=50)).exists()


def test_train_overwrite(model_dir, serializer):
    model_set1 = ModelSet(model_path=model_dir, serializer=serializer)
    runs = []

    def train_func(seed):
        runs.append(True)

    model_set1.train(train_func, seeds=3, require="sharedmem")
    assert len(runs) == 3

    runs.clear()
    assert len(runs) == 0
    model_set2 = ModelSet(model_path=model_dir, serializer=serializer)
    model_set2.train(train_func, seeds=3 + 2, require="sharedmem")
    assert len(runs) == 2


@pytest.mark.parametrize("verbose", [False, True])
def test_train_until_condition(model_dir, serializer, verbose):
    model_set = ModelSet(model_path=model_dir, serializer=serializer)
    score_values = [0.1] * 5 + [0.9] * 5 + [0.1] + [0.9] * 5
    train_func = lambda seed: score_values[seed]
    condition_func = lambda val: val > 0.5
    model_set.train_until_condition(
        train_func, condition_func, 6, max_seeds=len(score_values), verbose=verbose
    )
    assert sorted(model_set.get_seeds()) == list(range(12))
    condition_flags = model_set.apply(condition_func)
    assert sum(condition_flags) == 6


def test_seeds(model_dir, serializer):
    model_set = ModelSet(model_path=model_dir, serializer=serializer)
    train_func = lambda seed: "whatever"
    model_set.train(train_func, seeds=50)
    assert set(model_set.get_seeds()) == set(range(50))


@pytest.mark.parametrize("verbose", [False, True])
def test_apply(model_dir, serializer, verbose):
    model_set = ModelSet(model_path=model_dir, serializer=serializer)
    train_func = lambda seed: seed
    model_set.train(train_func, seeds=50)
    results = model_set.apply(lambda model: model, verbose=verbose)
    assert sum(results) == sum(range(50))


def test_apply_bad_seeds(model_dir, serializer):
    model_set = ModelSet(model_path=model_dir, serializer=serializer)
    train_func = lambda seed: seed
    model_set.train(train_func, seeds=50)
    with pytest.raises(ValueError):
        model_set.apply(lambda model: model, seeds=[51])


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_diabetes
from diffprivlib.models import LinearRegression
from diffprivlib.utils import PrivacyLeakWarning


class StubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def make_train_func(model_framework, X, y):
    if model_framework == "torch_module":
        return lambda seed: StubModule()
    elif model_framework == "sklearn":
        return lambda seed: LinearRegression(epsilon=1.0, random_state=seed).fit(X, y)


def make_eval_func(model_framework, X, y):
    if model_framework == "torch_module":
        return (
            lambda model: (model(torch.tensor(X).float()).detach().cpu().numpy() - y)
            ** 2
        )
    elif model_framework == "sklearn":
        return lambda model: model.score(X, y)


@pytest.mark.filterwarnings("ignore:::diffprivlib")
@pytest.mark.parametrize("model_framework", ["torch_module", "sklearn"])
def test_e2e(model_dir, model_framework, serializer):
    X, y = load_diabetes(return_X_y=True)
    model_set = ModelSet(model_path=model_dir)
    train_func = make_train_func(model_framework, X, y)
    eval_func = make_eval_func(model_framework, X, y)
    model_set.train(train_func=train_func, eval_func={"score": eval_func}, seeds=30)

    if model_framework == "torch_module":
        apply_func = lambda model: np.mean(
            (model(torch.tensor(X).float()).detach().numpy() - y) ** 2
        )
    else:
        apply_func = lambda model: np.mean((model.predict(X) - y) ** 2)

    scores = model_set.apply(apply_func)
    assert sum(scores) / len(scores) > 0
