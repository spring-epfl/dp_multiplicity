from argmapper.lib import parse_spec
from tabular_dp import get_data_splits


spec = parse_spec("exp/config/tabular_dp_experiments.yml")

for dataset in spec.parameters["dataset"]:
    X_train, X_test, y_train, y_test = get_data_splits(dataset)
    print(dataset)
    print(f"size={len(X_train) + len(X_test)}")
    print(f"{X_train.shape=} {X_test.shape=} {y_train.shape=} {y_test.shape=}\n")
