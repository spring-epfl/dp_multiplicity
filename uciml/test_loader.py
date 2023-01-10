import pytest
from .loader import load_dataset, targets


dataset_names = list(targets.keys())


@pytest.mark.parametrize("dataset_name", dataset_names)
@pytest.mark.parametrize("one_hot_encode", [False, True])
def test_load_dataset(dataset_name, one_hot_encode):
    X, y = load_dataset(dataset_name, one_hot_encoded=one_hot_encode)
    assert len(X) > 0
    assert len(X) == len(y)

    if one_hot_encode:
        assert len(list(X.select_dtypes(include=["object", "category"]))) == 0
