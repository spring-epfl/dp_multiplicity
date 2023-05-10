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

experiment_spec_path = pathlib.Path("../exp/config/tabular_dp_experiments.yml")
base_model_path = pathlib.Path("../out")
runs_data_backup_path = pathlib.Path("tabular_runs_data_backup.pkl")

wandb_entity = "bogdanspace/multiplicities"

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
import pickle
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
from sklearn.metrics import roc_auc_score, f1_score

from src.model_set import ModelSet
from exp.scripts.tabular_dp import get_data_splits

sns.set(style="white", context="paper", font_scale=1.75)


# %% [markdown]
# ## Load experiment metadata

# %%
def get_model_set_name(dataset, model, epsilon, **kwargs):
    if isinstance(epsilon, int):
        return f"{dataset}_{model}_{epsilon}.0"
    else:
        return f"{dataset}_{model}_{epsilon}"


# %%
spec = parse_spec(experiment_spec_path)

# %%
runs_data_cache = {}

# %%
if runs_data_backup_path.exists():
    with open(runs_data_backup_path, "rb") as f:
        runs_data = pickle.load(f)

else:
    for params in spec.expand():
        print(params)
        
        model_set_name = get_model_set_name(**params)
        if model_set_name in runs_data_cache:
            current_runs_data = runs_data_cache[model_set_name]
            print("Loading from cache")
        
        else:    
            runs = tracker.runs(wandb_entity, {
                "group": {"$eq": model_set_name},
            })

            summary_list = []
            config_list = []
            for run in tqdm.tqdm(runs):
                # .summary contains the output keys/values for metrics like accuracy.
                #  We call ._json_dict to omit large files 
                summary_list.append(run.summary._json_dict)

                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config_list.append(
                    {k: v for k,v in run.config.items()
                     if not k.startswith('_')})

            runs_data_cache[model_set_name] = pd.DataFrame({
                "summary": summary_list,
                "config": config_list,
            })

    runs_data = pd.concat(runs_data_cache.values())
    pd.to_pickle(runs_data, runs_data_backup_path)

runs_data.head()

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

# %% [markdown]
# ## Get model predictions and performance data

# %%
multiplicity_data = {}

# %%
perf_data = {}


# %%
def process_model(model_set, X, y):
    def get_test_outputs(model):
        return model.predict_proba(X)[:, 1]
    outputs = []
    perf = []
    for seed, preds in zip(model_set.get_seeds(), model_set.apply(get_test_outputs)):
        if preds is not None:
            for example_id, pred in enumerate(preds): 
                outputs.append(dict(seed=seed, pred=pred, y=y.iloc[example_id], example_id=example_id))
            
        perf.append(dict(
                seed=seed,
                f1=f1_score(y, preds > 0.5),
                auc=roc_auc_score(y, preds),
            )
        )
    
    return pd.DataFrame(outputs), pd.DataFrame(perf)
   
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
        outputs, perf = process_model(model_set, X_test, y_test)
        model_set_output = []
        for _, output_row in outputs.iterrows():
            rec = dict(model_set_name=model_set_name, **output_row)
            model_set_output.append(rec)
            
        multiplicity_data[model_set_name] = pd.DataFrame(model_set_output)
        perf_data[model_set_name] = perf


# %%
def viable_pred_range(multiplicity_data):
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


# %% [markdown]
# ## Evaluate disagreement metrics

# %%
metrics = {
    "pred_var": pred_var,
    "viable_pred_range": viable_pred_range,
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
     .query("0.5 < epsilon <= 2.5")
     .groupby(["model", "dataset", "epsilon", "example_id", "metric"])
     .value
     .mean()
     .reset_index()
     .pivot_table(index=["model", "dataset", "epsilon", "example_id"], columns="metric", values="value")
     .reset_index()
)

conf_data.loc[:, "confidence_vals"] = 0.5 + (conf_data.confidence_vals - 0.5).abs()
conf_data.loc[:, "confidence_vals"] = pd.cut(conf_data.confidence_vals, 5).apply(lambda val: float(val.mid))

# %% [markdown]
# ## Pick an example for an illustration

# %%
demo_dataset = "mammo"

# %%
X_train, X_test, y_train, y_test = get_data_splits(demo_dataset, norm=False)

# %%
low_eps_data = (
    mult_data.query(f"dataset == '{demo_dataset}' and epsilon == 0.5")
    .pivot_table(index="example_id", columns="metric", values="value")
)
high_eps_data = (
    mult_data.query(f"dataset == '{demo_dataset}' and epsilon == 2.5")
    .pivot_table(index="example_id", columns="metric", values="value")
)

# %%
# diff_example_ids = (low_eps_data.pred_var - high_eps_data.pred_var).sort_values().iloc[-30:].index
diff_example_ids = set(low_eps_data[(low_eps_data.pred_var >= 0.95)].index).intersection(
                       low_eps_data[(high_eps_data.pred_var < 0.6)].index)
diff_example_ids

# %%
pd.concat([high_eps_data.loc[diff_example_ids], low_eps_data.loc[diff_example_ids]], axis=1)

# %%
id_to_explore = 25
X_test.iloc[id_to_explore], y_test.iloc[id_to_explore]

# %%
high_eps_data.loc[id_to_explore], low_eps_data.loc[id_to_explore]

# %%
(multiplicity_data["mammo_lr_0.5"].query(f"example_id == {id_to_explore}").pred > 0.5).mean(), \
(multiplicity_data["mammo_lr_2.5"].query(f"example_id == {id_to_explore}").pred > 0.5).mean()

# %% [markdown]
# ## Plots

# %%
perf_data_ = pd.concat([v.assign(model_set_name=k) for k, v in perf_data.items()])
exp_metadata_ = (
    exp_metadata
    .loc[:, ["dataset", "model", "epsilon", "model_set_name"]]
    .drop_duplicates()
)

cols = ["dataset", "model", "epsilon", "model_set_name"]

combined_data = (
    pd.concat([
        (
            pd.concat(metrics_data.values())
            .merge(exp_metadata_, on=["model_set_name"], how="inner")
            .loc[:, cols + ["example_id", "metric_value", "metric_name"]]
            .rename(columns={
                "metric_value": "value",
                "metric_name": "metric"
            })
        ),
        (
            exp_metadata
            .merge(perf_data_, on=["model_set_name", "seed"])
            .query(f"model_set_name in {good_model_sets}")
            .query("0.5 <= epsilon <= 4")
            .loc[:, cols + ["seed", "f1", "auc", "beats_baseline", "top"]]
            .assign(auc=lambda df: df.auc * 100)
            .assign(f1=lambda df: df.f1 * 100)
            .melt(
                id_vars=["dataset", "model", "epsilon", "top", "beats_baseline"],
                value_vars=["auc", "f1"],
                var_name="metric",
            )
        )
    ])
    .fillna(True)
)

# %%
sns.relplot(
    data=(
        combined_data
        .query("model == 'lr'")
        .query("dataset in ['contrac', 'mammo', 'credit-approval', 'dermatology']")
        .query("metric in ['f1', 'auc', 'pred_var', 'viable_pred_range']")
        .query("0.5 <= epsilon <= 2.5")
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
            "auc": "AUC",
            "f1": "F1 score",
            "pred_var": "Disagreement",
            "good_pred_var": "Disagreement (beat baseline)",
            "top_pred_var": "Disagreement (top)",
            "viable_pred_range": "Viable pred. range"
        })
    ),
    kind="line",
    x=r"Privacy param. ε",
    y="Measure value",
    hue="Dataset",
    style="Dataset",
    col="Measure",
    marker="o",
    col_order=["AUC", "Disagreement"],
    hue_order=["Credit", "Dermatology", "Contraception", "Mammography"],
    facet_kws={'sharey': False, 'sharex': True},
)

plt.gca().invert_xaxis()
plt.savefig("../images/tabular_mult.pdf", bbox_inches="tight")

# %%
sns.relplot(
    data=(
        combined_data
        .query("model == 'lr'")
        .query("dataset in ['contrac', 'mammo', 'credit-approval', 'dermatology']")
        .query("metric in ['f1', 'auc', 'pred_var', 'viable_pred_range']")
        .query("0.5 <= epsilon <= 2.5")
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
            "auc": "AUC",
            "f1": "F1 score",
            "pred_var": "Disagreement",
            "good_pred_var": "Disagreement (beat baseline)",
            "top_pred_var": "Disagreement (top)",
            "viable_pred_range": "Viable pred. range"
        })
    ),
    kind="line",
    x=r"Privacy param. ε",
    y="Measure value",
    hue="Dataset",
    style="Dataset",
    col="Measure",
    marker="o",
    col_order=["Disagreement", "Viable pred. range"],
    hue_order=["Credit", "Dermatology", "Contraception", "Mammography"],
    facet_kws={'sharey': False, 'sharex': True},
)

plt.gca().invert_xaxis()
plt.savefig("../images/tabular_mult_vp.pdf", bbox_inches="tight")

# %%
g = sns.displot(
    data=(
        combined_data
        .query("model == 'lr' and dataset != 'ctg'")
        .query("0.5 <= epsilon <= 2.5")
        .query("metric in ['pred_var']")
#         .query("top == True")
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
            "pred_var": "Disagreement (all)",
        })
    ),
    x="Measure value",
    hue="Privacy param. ε",
    col="Dataset",
    kind="ecdf",
    palette="rocket",
    col_order=["Credit", "Dermatology", "Contraception", "Mammography"],
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
        .query("metric in ['pred_var']")
        .query("epsilon in [0.5, 1.0, 1.5, 2.0]")
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
            "pred_var": "Disagreement (all)",
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
    col_order=["Credit", "Dermatology", "Contraception", "Mammography"],
#     facet_kws={"sharex": False, "sharey": True},
#     sharex=False,
    palette="rocket",
    legend=False,
)

for ax in g.axes.flatten():
    ax.set_xlabel("Disagreement")
#     ax.set_xlim(ax.get_xlim()[0], 1)

plt.gca().invert_yaxis()

plt.savefig("../images/tabular_pdf.pdf", bbox_inches="tight")

# %%
dataset = "contrac"
attribute = "age"
X_train, X_test, y_train, y_test = get_data_splits(dataset, norm=False)
print(X_train.columns)
group_mult_data = (combined_data
    .query("metric == 'pred_var'")
    .query(f"dataset == '{dataset}'")
    .drop(columns=["metric"])
    .rename(columns={"value": "pred_var"})
    .merge(X_test.reset_index().loc[:, attribute], left_on="example_id", right_index=True)
)
group_mult_data.loc[:, f"{attribute}_disc"] = pd.cut(group_mult_data.loc[:, attribute], [16, 30, 40, 50])
# group_mult_data.loc[:, f"{attribute}_disc"] = pd.qcut(group_mult_data.loc[:, attribute], 4)
group_mult_data

# %%
temp_eps_vals = [2.0, 1.5, 1.0, 0.5]

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
            "age_disc": "Age",
            "pred_var": "Disagreement",
        })
    ),
    kind="bar",
    x="Disagreement",
    y="Age",
    col="Privacy param. ε",
    col_order=temp_eps_vals,
    palette="rocket_r",
    height=3,
    aspect=1.33,
)
plt.savefig("../images/tabular_disparity.pdf", bbox_inches="tight")

# %%
(group_mult_data
     .query(f"epsilon in {temp_eps_vals}")
     .groupby(["dataset", "epsilon", "age_disc"])
     .pred_var.mean()
)

# %%
temp_eps_vals = [2.5, 2.0, 1.5, 1.25, 1.0, 0.5]
sns.relplot(
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
            "pred_var": "Disagreement",
        })
    ),
    kind="scatter",
    x="Age",
    y="Disagreement",
    col="Privacy param. ε",
    palette="rocket_r"
)

# %% [markdown]
# ## Table

# %%
auc_data = (
    combined_data
    .query("metric in ['auc', 'f1']")
    .query("0.5 <= epsilon <= 2.5")
    .pivot_table(
        index=["dataset", "epsilon"], columns="metric", values="value",
        aggfunc=("mean", "std"),
    )
)

pred_var_data = (
    combined_data
    .query("metric in ['pred_var']")
    .query("0.5 <= epsilon <= 2.5")
    .pivot_table(
        index=["dataset", "epsilon"], columns="metric", values="value",
        aggfunc=("mean", "std", "median", "min", "max",
                 lambda df: np.percentile(df, 90), lambda df: np.percentile(df, 95)),
    )
)[["mean", "std", "min", "median", "max", "<lambda_0>", "<lambda_1>"]].rename(
    columns={"<lambda_0>": "90pctl", "<lambda_1>": "95pctl"})

print(
    pd.concat([auc_data, pred_var_data], axis=1)
    .round(2)
    .reorder_levels([1, 0], axis=1)[["auc", "f1", "pred_var"]]
    .to_latex()
)
