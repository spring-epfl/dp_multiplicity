# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pathlib
import itertools
import collections

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from argmapper import parse_spec
from tqdm import autonotebook as tqdm
from matplotlib import pyplot as plt

from src.model_set import ModelSet
from exp.cifar10 import convnet

sns.set(style="white", context="paper", font_scale=1.75)

# %%
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

# %%
experiment_spec_path = pathlib.Path("../exp/config/cifar10_dp_experiments.yml")
base_model_path = pathlib.Path("../out")


# %%
def get_model_set_name(dataset, model, epsilon):
    if isinstance(epsilon, int):
        return f"{dataset}_{model}_{epsilon}.0"
    else:
        return f"{dataset}_{model}_{epsilon}"


# %%
summary_list, config_list, name_list = [], [], []

# %%
import wandb
tracker = wandb.Api()
runs = tracker.runs(f"bogdanspace/multiplicities", {"config.dataset": {"$eq": "cifar10"}})

for run in tqdm.tqdm(runs):        
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_data = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
})

runs_data.head()

# %%
exp_cached_metadata = {}
baseline = 0.1

for _, run in runs_data.iterrows():
    timestamp = run.summary.get("_timestamp")
    if not timestamp:
        continue
        
    run_config_hash = str(dict(sorted(run.config.items())))
    if run_config_hash in exp_cached_metadata:
        prev_timestamp = exp_cached_metadata[run_config_hash]["timestamp"]
        if prev_timestamp > timestamp:
            continue
        
    run_metadata = {
        "timestamp": timestamp,
        "test_acc": run.summary.get("test_acc"),
        "normalized_adv": 2 * (run.summary.get("test_acc") - baseline),
        "beats_baseline": run.summary.get("test_acc") > baseline,
#         "seed": run.config.get("seed"),
        "dataset": run.config.get("dataset"),
        "model": run.config.get("model"),
        "epsilon": run.summary.get("epsilon"),
        "model_set_name": run.config.get("name"),
    }
    exp_cached_metadata[run_config_hash] = run_metadata
    
exp_metadata = pd.DataFrame(exp_cached_metadata.values())
exp_metadata

# %%
spec = parse_spec(experiment_spec_path)


# %%
class TorchSerializer:
    def load(self, f):
        print(f.name)
        state = torch.load(f)
        if state["model"] == "simple":
            model = convnet(num_classes=10)
            state_dict = {k.lstrip("_module."): v for k, v in state["state_dict"].items()}
            model.load_state_dict(state_dict)
        return model

# Test.
with open("../out/cifar10_simple_0.5/model_0", "rb") as f:
    loaded_model = TorchSerializer().load(f)

# %%
model_sets = {}

for model_set_name in exp_metadata.model_set_name.unique():
    model_set_path = base_model_path / model_set_name
    model_set = ModelSet(model_set_path, serializer=TorchSerializer())
    try:
        assert len(model_set) > 0
        model_sets[model_set_name] = model_set
    except FileNotFoundError:
        print(f"{model_set_path} not found.")
        
len(model_sets)

# %%
multiplicity_data = {}

# %%
normalize = [
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]
test_transform = transforms.Compose(normalize)
test_dataset = CIFAR10("../../cifar10", train=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)


# %%
def process_model(model_set, loader):    
    def get_test_outputs(model):
        preds = []
        for images, target in loader:
            out = list(model(images).detach().cpu().numpy())
            preds.extend(out)
        return preds

    outputs = []
    for seed, preds in zip(model_set.get_seeds(), model_set.apply(get_test_outputs)):
        if preds is not None:
            for example_id, pred in enumerate(preds):
                outputs.append(dict(seed=seed, pred=pred, example_id=example_id))
    
    return pd.DataFrame(outputs)


model_sets_to_process = exp_metadata.model_set_name.unique()

for model_set_name in tqdm.tqdm(model_sets_to_process):
    if model_set_name in multiplicity_data:
        continue

    if model_set_name not in model_sets:
        print(f"{model_set_name} not in loaded model sets.")
        continue
        
    model_set = model_sets[model_set_name]
    outputs = process_model(model_set, test_loader)
    model_set_output = []
    for _, output_row in outputs.iterrows():
        rec = dict(model_set_name=model_set_name, **output_row)
        model_set_output.append(rec)

    multiplicity_data[model_set_name] = pd.DataFrame(model_set_output)


# %%
def class_disc(multiplicity_data):
    results = []
    for example_id in multiplicity_data.example_id.unique():
        example_outputs = np.array([
            vec for vec in multiplicity_data.query(
            f"example_id == {example_id}").pred
        ])
        for class_idx in range(example_outputs.shape[1]):
            metric = example_outputs[:, class_idx].max() \
                   - example_outputs[:, class_idx].min()
            results.append(
                dict(
                    example_id=example_id,
                    class_idx=class_idx,
                    metric_value=metric,
                )
            )
    return pd.DataFrame(results)

def class_var(multiplicity_data):
    results = []
    for example_id in multiplicity_data.example_id.unique():
        example_outputs = np.array([
            vec for vec in multiplicity_data.query(
            f"example_id == {example_id}").pred
        ])
        for class_idx in range(example_outputs.shape[1]):
            metric = example_outputs[:, class_idx].var()
            results.append(
                dict(
                    example_id=example_id,
                    class_idx=class_idx,
                    metric_value=metric,
                )
            )
    return pd.DataFrame(results)

def binarized_var(multiplicity_data):
    results = []
    for example_id in multiplicity_data.example_id.unique():
        example_outputs = np.array([
            vec for vec in multiplicity_data.query(
            f"example_id == {example_id}").pred
        ])
        preds = example_outputs.argmax(axis=1)
        assert len(preds) == len(example_outputs)
        for class_idx in range(example_outputs.shape[1]):
            binarized_preds = (preds == class_idx)
            metric = 4 * binarized_preds.var()
            results.append(
                dict(
                    example_id=example_id,
                    class_idx=class_idx,
                    metric_value=metric,
                )
            )
    return pd.DataFrame(results)


# %%
metrics = {
#     "class_var": class_var,
    "binarized_var": binarized_var,
}

# %%
metrics_data = collections.defaultdict(list)

# %%
for model_set_name, model_set in tqdm.tqdm(list(model_sets.items())):
    if model_set_name not in multiplicity_data:
        continue
        
    if model_set_name in metrics_data:
        continue
        
    metrics_output = []
    for metric_name, metric_func in metrics.items():
        metrics_output.append(
            metric_func(multiplicity_data[model_set_name]).assign(
                metric_name=metric_name, model_set_name=model_set_name,
            )
        )
    
    metrics_data[model_set_name] = pd.concat(metrics_output)

# %%
exp_stats = (
    exp_metadata.groupby(["dataset", "model", "epsilon", "model_set_name"])
    .agg(dict(test_acc="mean"))
    .reset_index()
)

plot_data = (
    pd.concat(metrics_data.values())
    .merge(exp_stats, on="model_set_name", how="inner")
)
plot_data

# %%
combined_data = pd.concat([
    (
        plot_data
#         .query("metric_name == 'var'")
        .rename(columns={"metric_value": "value", "metric_name": "metric"})
        .drop(columns=["test_acc", "example_id"])
    ),
    (
        exp_metadata
        .query(f"model_set_name in {list(plot_data.model_set_name.unique())}")
        [
            ["dataset", "model", "epsilon", "model_set_name", "test_acc"]
        ]
        .assign(value=lambda df: 2 * (df["test_acc"] - 0.1))
        .assign(metric="normalized_adv")
    )
]).assign(epsilon=lambda df: df["epsilon"].round(4))

# Normalize.
def normalize_metric(data_slice):
    baseline = data_slice.query(f"epsilon == {data_slice.epsilon.min()}").value.mean()
    result = data_slice.copy()
    result["value"] = (data_slice["value"] / baseline - 1) * 100
    return result

normalized_combined_data = (
    combined_data
    .groupby(["dataset", "model", "metric"], group_keys=False)
    .apply(normalize_metric)
    .reset_index()
)

# %%
sorted(combined_data.epsilon.unique())

# %%
g = sns.relplot(
    data=(
        combined_data
        .rename(columns={
            "metric": "Measure",
            "value": "Measure value",
            "epsilon": "Privacy param. ε",
        })
        .replace({
            "normalized_adv": "Normalized accuracy",
            "binarized_var": "Disagreement",
        })
    ),
    kind="line",
    x=r"Privacy param. ε",
    y="Measure value",
#     hue="class_idx",
    col="Measure",
    marker="o",
    err_style="bars",
    col_order=["Normalized accuracy", "Disagreement"],
    facet_kws={'sharey': False, 'sharex': True}
)

plt.gca().invert_xaxis()
for ax in g.axes[0]:
    ax.set_title("")
# plt.savefig("../images/cifar10_mult.pdf", bbox_inches="tight")

# %%
g = sns.relplot(
    data=(
        combined_data
        .rename(columns={
            "metric": "Measure",
            "value": "Measure value",
            "epsilon": "Privacy param. ε",
        })
        .replace({
            "binarized_var": "Disagreement",
        })
    ),
    kind="line",
    x=r"Privacy param. ε",
    y="Measure value",
    hue="class_idx",
    marker="o",
    err_style="bars",
    facet_kws={'sharey': False, 'sharex': True},
    legend=False,
)

plt.gca().invert_xaxis()
# plt.savefig("../images/cifar10_disparities.pdf", bbox_inches="tight")
