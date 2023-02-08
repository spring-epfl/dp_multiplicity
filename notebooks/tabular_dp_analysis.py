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
from exp.tabular_dp import get_data_splits

sns.set(style="white", context="paper", font_scale=1.75)

# %%
experiment_spec_path = pathlib.Path("../exp/config/tabular_dp_experiments.yml")
base_model_path = pathlib.Path("../out")
spec = parse_spec(experiment_spec_path)


# %%
def get_model_set_name(dataset, model, epsilon):
    if isinstance(epsilon, int):
        return f"{dataset}_{model}_{epsilon}.0"
    else:
        return f"{dataset}_{model}_{epsilon}"


# %%
summary_list, config_list, id_list = [], [], []

import wandb
tracker = wandb.Api()
runs = tracker.runs(f"bogdanspace/multiplicities", {
    "config.dataset": {"$in": spec.parameters["dataset"]},
    "config.model": {"$in": spec.parameters["model"]}
})

for run in tqdm.tqdm(list(runs)):
    if run.id in id_list:
        continue
      
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    id_list.append(run.id)

runs_data = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
})

runs_data.head()

# %%
# pd.to_pickle(runs_data, "runs_data_backup.pkl")

# %%
# runs_data = pd.read_pickle("runs_data_backup.pkl")

# %%
exp_cached_metadata = {}

for _, run in runs_data.iterrows():
    timestamp = run.summary.get("_timestamp")
    if not timestamp:
        continue
        
    if run.config["dataset"] == "cifar10":
        continue

    run_config_hash = str(dict(sorted(run.config.items())))
    if run_config_hash in exp_cached_metadata:
        prev_timestamp = exp_cached_metadata[run_config_hash]["timestamp"]
        if prev_timestamp > timestamp:
            continue
    
    run_metadata = {
        "timestamp": timestamp,
        "test_acc": run.summary.get("test_acc"),
        "normalized_adv": run.summary.get("normalized_adv"),
        "beats_baseline": run.summary.get("beats_baseline"),
        "seed": run.config.get("seed"),
        "dataset": run.config.get("dataset"),
        "model": run.config.get("model"),
        "epsilon": run.config.get("epsilon"),
    }
    
    model_set_name = get_model_set_name(
        dataset=run_metadata["dataset"],
        model=run_metadata["model"],
        epsilon=run_metadata["epsilon"],
    )
    model_set_path = base_model_path / model_set_name
    run_metadata["model_set_path"] = model_set_path
    run_metadata["model_set_name"] = model_set_name
    exp_cached_metadata[run_config_hash] = run_metadata
    
exp_metadata = pd.DataFrame(exp_cached_metadata.values())
acc_threshold = exp_metadata.groupby("model_set_name").normalized_adv.quantile(.25)
exp_metadata.loc[:, "top"] = exp_metadata.apply(
    lambda row: row.normalized_adv > acc_threshold.loc[row.model_set_name],
    axis=1
)

del runs_data
exp_metadata.head()

# %%
print(exp_metadata
      .query("model == 'lr'")
      .query(f"dataset in {spec.parameters['dataset']}")
      .groupby(["dataset", "model", "epsilon"])
      .seed
      .count()
      .to_string()
)

# %%
print(exp_metadata
      .query("model == 'lr'")
      .query(f"dataset in {spec.parameters['dataset']}")
      .groupby(["dataset", "model", "epsilon"])
      .top
      .sum()
      .to_string()
)

# %%
print(exp_metadata
      .query("model == 'lr'")
      .query(f"dataset in {spec.parameters['dataset']}")
      .groupby(["dataset", "model", "epsilon"])
      .beats_baseline
      .sum()
      .to_string()
)

# %%
good_model_sets = list(exp_metadata
      .query("model == 'lr'")
      .query(f"dataset in {spec.parameters['dataset']}")
      .groupby("model_set_name")
      .beats_baseline
      .sum()
      .loc[lambda x: x >= 100]
      .index
)
len(good_model_sets)

# %%
model_sets = {}

for model_set_name in good_model_sets:
    model_set_path = base_model_path / model_set_name
    model_set = ModelSet(model_set_path)
    try:
        assert len(model_set) > 0
        model_sets[model_set_name] = model_set
    except FileNotFoundError:
        print(f"{model_set_path} not found.")
        
len(model_sets)

# %%
multiplicity_data = {}


# %%
def process_model(model_set, X, y):
    baseline = max(y.mean(), 1 - y.mean())
    
    def get_test_outputs(model):
        return model.predict_proba(X)[:, 1]

    outputs = []
    for seed, preds in zip(model_set.get_seeds(), model_set.apply(get_test_outputs)):
        if preds is not None:
            for example_id, pred in enumerate(preds):
                outputs.append(dict(seed=seed, pred=pred, example_id=example_id))
    
    return pd.DataFrame(outputs)
   
datasets = spec.parameters["dataset"]
for dataset in datasets:
    print(dataset)
    X_train, X_test, y_train, y_test = get_data_splits(dataset)
    dataset_model_sets = set(exp_metadata.query(f"dataset == '{dataset}'").model_set_name.unique())
    model_sets_to_process = dataset_model_sets.intersection(good_model_sets)
    
    for model_set_name in tqdm.tqdm(model_sets_to_process):
        if model_set_name in multiplicity_data:
            continue
        
        if model_set_name not in model_sets:
            print(f"{model_set_name} not in loaded model sets.")
            continue
            
        model_set = model_sets[model_set_name]
        outputs = process_model(model_set, X_test, y_test)
        model_set_output = []
        for _, output_row in outputs.iterrows():
            rec = dict(model_set_name=model_set_name, **output_row)
            model_set_output.append(rec)
            
        multiplicity_data[model_set_name] = pd.DataFrame(model_set_output)


# %%
def discrepancy(multiplicity_data):
    results = []
    for example_id in multiplicity_data.example_id.unique():
        example_outputs = multiplicity_data.query(f"example_id == {example_id}").pred
        metric = example_outputs.max() - example_outputs.min()
        results.append(dict(example_id=example_id, seed=None, metric_value=metric))
    return pd.DataFrame(results)

def confidence_var(multiplicity_data):
    results = []
    for example_id in multiplicity_data.example_id.unique():
        example_outputs = (multiplicity_data.query(f"example_id == {example_id}").pred)
        metric = 4 * example_outputs.var()
        results.append(dict(example_id=example_id, seed=None, metric_value=metric))
    return pd.DataFrame(results)

def pred_var(multiplicity_data):
    results = []
    for example_id in multiplicity_data.example_id.unique():
        example_outputs = (multiplicity_data.query(f"example_id == {example_id}").pred > 0.5)
        metric = 4 * example_outputs.var()
        results.append(dict(example_id=example_id, seed=None, metric_value=metric))
    return pd.DataFrame(results)

def good_pred_var(multiplicity_data):
    results = []
    model_set_name = multiplicity_data.model_set_name.unique()[0]
    top_seeds = list(
        exp_metadata.query(f"model_set_name == '{model_set_name}' and beats_baseline == True").seed.dropna()
    )
    if not top_seeds:
        return
    
    top_data = multiplicity_data.query(f"seed in {top_seeds}")
    for example_id in multiplicity_data.example_id.unique():
        example_outputs = (top_data.query(f"example_id == {example_id}").pred > 0.5)
        metric = 4 * example_outputs.var()
        results.append(dict(example_id=example_id, seed=None, metric_value=metric))
    return pd.DataFrame(results)

def top_pred_var(multiplicity_data):
    results = []
    model_set_name = multiplicity_data.model_set_name.unique()[0]
    top_seeds = list(
        exp_metadata.query(f"model_set_name == '{model_set_name}' and top == True").seed.dropna()
    )
    if not top_seeds:
        return
    
    top_data = multiplicity_data.query(f"seed in {top_seeds}")
    for example_id in multiplicity_data.example_id.unique():
        example_outputs = (top_data.query(f"example_id == {example_id}").pred > 0.5)
        metric = 4 * example_outputs.var()
        results.append(dict(example_id=example_id, seed=None, metric_value=metric))
    return pd.DataFrame(results)

def confidence_vals(multiplicity_data):
    return multiplicity_data.loc[:, ["example_id", "seed", "pred"]].rename(
        columns={"pred": "metric_value"}
    )


# %%
metrics = {
    "pred_var": pred_var,
#     "good_pred_var": good_pred_var,
    "top_pred_var": top_pred_var,
    "confidence_vals": confidence_vals,
}

# %%
metrics_data = {}

# %%
for model_set_name, model_set in tqdm.tqdm(list(model_sets.items())):
    if model_set_name not in multiplicity_data:
        continue
        
    if model_set_name in metrics_data:
        continue
        
    metrics_output = []
    for metric_name, metric_func in metrics.items():
        metric_vals = metric_func(multiplicity_data[model_set_name])
        if metric_vals is not None:
            metrics_output.append(metric_vals.assign(
                metric_name=metric_name,
                model_set_name=model_set_name
            ))
    
    metrics_data[model_set_name] = pd.concat(metrics_output)

# %%
exp_stats = (
    exp_metadata
    .loc[:, ["dataset", "model", "epsilon", "model_set_name"]]
    .drop_duplicates()
)

cols = ["dataset", "model", "epsilon", "model_set_name"]

mult_data = (
    (
        pd.concat(metrics_data.values())
        .merge(exp_stats, on=["model_set_name"], how="inner")
        .loc[:, cols + ["example_id", "metric_value", "metric_name"]]
        .rename(columns={
            "metric_value": "value",
            "metric_name": "metric"
        })
    )
    .fillna(True)
)

# %%
conf_data = (mult_data
     .query("1 < epsilon <= 5.0")
     .groupby(["model", "dataset", "epsilon", "example_id", "metric"])
     .value
     .mean()
     .reset_index()
     .pivot_table(index=["model", "dataset", "epsilon", "example_id"], columns="metric", values="value")
     .reset_index()
)

conf_data.loc[:, "confidence_vals"] = 0.5 + (conf_data.confidence_vals - 0.5).abs()
conf_data.loc[:, "confidence_vals"] = pd.cut(conf_data.confidence_vals, 5).apply(lambda val: float(val.mid))

# %%
X_train, X_test, y_train, y_test = get_data_splits("mammo")

# %%
low_eps_data = (
    mult_data.query("dataset == 'mammo' and epsilon == 0.5")
    .pivot_table(index="example_id", columns="metric", values="value")
#     .set_index("example_id")
)
high_eps_data = (
    mult_data.query("dataset == 'mammo' and epsilon == 4.0")
    .pivot_table(index="example_id", columns="metric", values="value")
)

diff_example_ids = (low_eps_data.pred_var - high_eps_data.pred_var).sort_values().iloc[-10:].index

# %%
pd.concat([high_eps_data.loc[diff_example_ids], low_eps_data.loc[diff_example_ids]], axis=1)

# %%
X_test.iloc[99], y_test.iloc[99]

# %%
low_eps_data.loc[99], high_eps_data.loc[99]

# %%
exp_stats = (
    exp_metadata
    .loc[:, ["dataset", "model", "epsilon", "model_set_name"]]
    .drop_duplicates()
)

cols = ["dataset", "model", "epsilon", "model_set_name"]

combined_data = (
    pd.concat([
        (
            pd.concat(metrics_data.values())
            .merge(exp_stats, on=["model_set_name"], how="inner")
            .loc[:, cols + ["example_id", "metric_value", "metric_name"]]
            .rename(columns={
                "metric_value": "value",
                "metric_name": "metric"
            })
        ),
        (
            exp_metadata
            .query(f"model_set_name in {good_model_sets}")
            .query("epsilon <= 20")
            .loc[:, cols + ["normalized_adv", "beats_baseline", "top"]]
            .rename(columns={"normalized_adv": "value"})
            .assign(metric="normalized_adv")
        )
    ])
    .fillna(True)
)

# # Normalize.
# def normalize_metric(data_slice):
#     baseline = data_slice.query(f"epsilon == {data_slice.epsilon.min()}").value.mean()
#     result = data_slice.copy()
#     result["value"] = (data_slice["value"] / baseline - 1) * 100
#     return result

# normalized_combined_data = (
#     combined_data
#     .groupby(["dataset", "model", "metric"], group_keys=False)
#     .apply(normalize_metric)
#     .reset_index()
# )

# %%
sns.relplot(
    data=(
        combined_data
        .query("model == 'lr'")
        .query("1 < epsilon <= 5.0")
        .query("top == True")
        .rename(columns={
            "dataset": "Dataset",
            "model": "Model",
            "metric": "Measure",
            "value": "Measure value",
            "epsilon": "Privacy param. ε",
        })
        .replace({
            "credit-approval": "Credit",
            "contrac": "Contraception",
            "mammo": "Mammography",
            "dermatology": "Dermatology",
            "normalized_adv": "Normalized accuracy",
            "pred_var": "Disagreement (all)",
            "good_pred_var": "Disagreement (beat baseline)",
            "top_pred_var": "Disagreement",
        })
    ),
    kind="line",
    x=r"Privacy param. ε",
    y="Measure value",
    hue="Dataset",
    style="Dataset",
    col="Measure",
    marker="o",
    col_order=["Normalized accuracy", "Disagreement"],
    hue_order=["Credit", "Contraception", "Mammography", "Dermatology"],
    facet_kws={'sharey': False, 'sharex': True},
)

plt.gca().invert_xaxis()
plt.savefig("../images/tabular_mult.pdf", bbox_inches="tight")

# %%
g = sns.displot(
    data=(
        combined_data
        .query("model == 'lr' and dataset != 'ctg'")
        .query("1 < epsilon < 5")
        .query("metric == 'top_pred_var'")
        .query("top == True")
        .rename(columns={
            "dataset": "Dataset",
            "model": "Model",
            "metric": "Measure",
            "value": "Measure value",
            "epsilon": "Privacy param. ε",
        })
        .replace({
            "credit-approval": "Credit",
            "contrac": "Contraception",
            "mammo": "Mammography",
            "dermatology": "Dermatology",
            "normalized_adv": "Normalized accuracy",
            "top_pred_var": "Disagreement",
        })
    ),
    x="Measure value",
    hue="Privacy param. ε",
    col="Dataset",
    kind="ecdf",
    palette="rocket",
    col_order=["Credit", "Contraception", "Mammography", "Dermatology"],
    facet_kws={'sharey': True, 'sharex': False},
    legend=False,
)

for ax in g.axes.flatten():
    ax.set_xlabel("Disagreement")
    
norm = plt.Normalize(1, 5)
sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=g.axes.flatten()[-1])
cbar.set_label("Privacy param. ε")
plt.savefig("../images/tabular_ecdf.pdf", bbox_inches="tight")

# %%
g = sns.catplot(
    data=(
        combined_data
        .query("model == 'lr'")
        .query("metric == 'top_pred_var'")
        .query("top == True")
        .query("epsilon in [1.0, 2.0, 3.0, 4.0]")
        .rename(columns={
            "dataset": "Dataset",
            "model": "Model",
            "metric": "Measure",
            "value": "Measure value",
            "epsilon": "Privacy param. ε",
        })
        .replace({
            "credit-approval": "Credit",
            "contrac": "Contraception",
            "mammo": "Mammography",
            "dermatology": "Dermatology",
            "normalized_adv": "Normalized accuracy",
            "top_pred_var": "Disagreement",
        })
    ),
    y="Privacy param. ε",
    x="Measure value",
    col="Dataset",
#     hue="Privacy param. ε",
    kind="violin",
    bw=0.15,
    cut=0.5,
    split=True,
    hue=True,
    hue_order=[False, True],
    inner="quartile",
    orient="h",
    col_order=["Credit", "Contraception", "Mammography", "Dermatology"],
#     facet_kws={"sharex": False, "sharey": True},
    palette="rocket",
    legend=False,
)

for ax in g.axes.flatten():
    ax.set_xlabel("Disagreement")
#     ax.set_xlim(ax.get_xlim()[0], 1)

plt.gca().invert_yaxis()

plt.savefig("../images/tabular_pdf.pdf", bbox_inches="tight")

# %%
acc_levels = np.linspace(0, 0.02, 10)
acc_level_data = []
it = list(itertools.product(
    spec.parameters["dataset"],
    spec.parameters["model"],
    spec.parameters["epsilon"],
))

for dataset, model, epsilon in tqdm.tqdm(it):
    exp_slice = (exp_metadata
        .query(f"dataset == '{dataset}'")
        .query(f"model == '{model}'")
        .query(f"epsilon == {epsilon}")
    )
    max_acc = exp_slice.normalized_adv.max()
    if np.isnan(max_acc):
        continue
    
    for acc_level in acc_levels:
        acc_slice = exp_slice.query(f"normalized_adv >= {max_acc} - {acc_level}")
        model_set_name = acc_slice.model_set_name.unique()[0]
        seeds = list(acc_slice.seed.dropna().unique())
        if not seeds:
            continue
        
        dis_vals = pred_var(multiplicity_data[model_set_name].query(f"seed in {seeds}")).fillna(0)
        acc_level_data.append(
            (pd.DataFrame(dict(disagreement=dis_vals.metric_value))
                .assign(
                    acc_level=acc_level,
                    epsilon=epsilon,
                    dataset=dataset,
                    model=model,
                    model_set_name=model_set_name,
                )
            )
        )

# %%
sns.relplot(
    data=pd.concat(acc_level_data),
    x="acc_level",
    y="disagreement",
    hue="epsilon",
    col="dataset",
    kind="line",
    errorbar=None,
    err_style="bars",
#     estimator=lambda s: np.quantile(s, .95)
)

# %%
dataset = "mammo"
attribute = "age"
X_train, X_test, y_train, y_test = get_data_splits(dataset)
group_mult_data = (combined_data
    .query("metric == 'top_pred_var'")
    .query(f"dataset == '{dataset}'")
    .drop(columns=["metric"])
    .rename(columns={"value": "top_pred_var"})
    .merge(X_test.reset_index().loc[:, attribute], left_on="example_id", right_index=True)
)
group_mult_data.loc[:, attribute] = pd.qcut(group_mult_data.loc[:, attribute], 4)
group_mult_data

# %%
temp_eps_vals = [5.0, 2.5, 1.0]
sns.catplot(
    data=(
        group_mult_data
        .query(f"epsilon in {temp_eps_vals}")
        .rename(columns={
            "dataset": "Dataset",
            "model": "Model",
            "metric": "Measure",
            "value": "Measure value",
            "epsilon": "Privacy param. ε",
            "normalized_adv": "Normalized accuracy",
            "age": "Age",
            "top_pred_var": "Disagreement",
        })
    ),
    kind="bar",
    x="Disagreement",
    y="Age",
    col="Privacy param. ε",
    col_order=temp_eps_vals,
    palette="rocket_r"
)
plt.savefig("../images/tabular_disparity.pdf", bbox_inches="tight")
