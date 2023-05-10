# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pathlib

handcrafted_dp_path = pathlib.Path("../ext/Handcrafted-DP")
bn_stats_save_dir = pathlib.Path("../bn_stats/cifar10")
experiment_spec_path = pathlib.Path("../exp/config/cifar10_dp_experiments.yml")
base_model_path = pathlib.Path("../out")
runs_data_path = pathlib.Path("cifar10_runs_data_backup.pkl")

wandb_entity = "bogdanspace/multiplicities"
device = "cuda:1"

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys
import pickle
import itertools
import collections

# %%
sys.path.append(str(handcrafted_dp_path))

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from argmapper import parse_spec
from tqdm import autonotebook as tqdm
from matplotlib import pyplot as plt

from src.model_set import ModelSet
from data import get_scattered_loader, get_scatter_transform, get_data
from models import CNNS

sns.set(style="white", context="paper", font_scale=1.75)

# %%
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

# %% [markdown]
# ## Get experiment metadata

# %%
if runs_data_path.exists():
    with open(runs_data_path, "rb") as f:
        runs_data = pickle.load(f)
        
else:
    summary_list, config_list, name_list = [], [], []
    
    import wandb
    tracker = wandb.Api()
    runs = tracker.runs(wandb_entity, {"config.dataset": {"$eq": "cifar10"}})

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

    with open(runs_data_path, "wb") as f:
        pickle.dump(runs_data, f)
        
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
        "beats_baseline": run.summary.get("test_acc") > baseline,
        "epochs": run.summary.get("epoch"),
        "seed": run.config.get("seed"),
        "sigma": run.config.get("sigma"),
        "dataset": run.config.get("dataset"),
        "epsilon": run.summary.get("epsilon"),
        "model_set_name": run.config.get("name"),
    }
    exp_cached_metadata[run_config_hash] = run_metadata
    
exp_metadata = pd.DataFrame(exp_cached_metadata.values())
exp_metadata

# %% [markdown]
# ## Retrieve predictions

# %%
spec = parse_spec(experiment_spec_path)

# %%
# Reproduce the data loaders.
mini_batch_size = 256

train_data, test_data = get_data("cifar10", augment=False)
scattering, K, _ = get_scatter_transform("cifar10")

raw_test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=mini_batch_size, shuffle=False, num_workers=1, pin_memory=True,
)
test_loader = get_scattered_loader(raw_test_loader, scattering.to(device), device=device)

# %%
# Reproduce the data pre-computed batch norm stats.
noise_multiplier = 8.0

mean_path = os.path.join(bn_stats_save_dir, f"mean_bn_{len(train_data)}_{noise_multiplier}_True.npy")
var_path = os.path.join(bn_stats_save_dir, f"var_bn_{len(train_data)}_{noise_multiplier}_True.npy")

mean = torch.from_numpy(np.load(mean_path)).to(device)
var = torch.from_numpy(np.load(var_path)).to(device)
bn_stats = (mean, var)


# %%
class TorchSerializer:
    def load(self, f):
        print(f.name)
        state = torch.load(f)
        model = CNNS["cifar10"](K, input_norm="BN", bn_stats=bn_stats, size=None).to(device)
        model.load_state_dict(state["state_dict"])
        return model

# Test deserialization.
with open("../out/cifar10_scatternet_4.0/model_49", "rb") as f:
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
# with open("cifar10_multiplicity_data_backup.pkl", "rb") as f:
#     multiplicity_data = pickle.load(f)

# %%
multiplicity_data = {}


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


# %% [markdown]
# ## Evaluate disagreement metrics

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

# %% [markdown]
# ## Plots

# %%
exp_stats = (
    exp_metadata.groupby(["dataset", "model_set_name"])
    .agg(dict(test_acc="mean", epsilon="max"))
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
        .rename(columns={"metric_value": "value", "metric_name": "metric"})
        .drop(columns=["test_acc", "example_id"])
    ),
    (
        exp_metadata
        .query(f"model_set_name in {list(plot_data.model_set_name.unique())}")
        .assign(epsilon=lambda df: exp_stats.set_index("model_set_name").loc[df.model_set_name].epsilon.values)
        [
            ["dataset", "epsilon", "model_set_name", "test_acc"]
        ]
        .rename(columns={"test_acc": "value"})
        .assign(metric="test_acc")
    )
]).assign(epsilon=lambda df: df["epsilon"].round(4))

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
            "test_acc": "Test accuracy",
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
    errorbar=("ci", 99),
    col_order=["Test accuracy", "Disagreement"],
    facet_kws={'sharey': False, 'sharex': True}
)

plt.xticks([6, 5, 4, 3, 2])
plt.gca().invert_xaxis()
# for ax in g.axes[0]:
#     ax.set_title("")
plt.savefig("../images/cifar10_mult.pdf", bbox_inches="tight")

# %% [markdown]
# ## Table

# %%
auc_data = (
    combined_data
    .query("metric in ['test_acc']")
    .pivot_table(
        index=["dataset", "epsilon"], columns="metric", values="value",
        aggfunc=("mean", "std"),
    )
)

pred_var_data = (
    combined_data
    .query("metric in ['binarized_var']")
    .assign(epsilon=lambda df: df.epsilon.astype(float))
    .pivot_table(
        index=["dataset", "epsilon"], columns="metric", values="value",
        aggfunc=("mean", "std", "median", "min", "max",
                 lambda df: np.percentile(df, 90), lambda df: np.percentile(df, 95)),
    )
)[["mean", "std", "min", "median", "max", "<lambda_0>", "<lambda_1>"]].rename(
    columns={"<lambda_0>": "90pctl", "<lambda_1>": "95pctl"})

print(pd.concat([auc_data, pred_var_data], axis=1).round(2).to_latex())
