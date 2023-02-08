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
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import diffprivlib
import seaborn as sns
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from tqdm import autonotebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn.inspection import DecisionBoundaryDisplay as DBD
from scipy.optimize import root
from scipy.stats import norm
from matplotlib import animation

sns.set(style="white", context="paper", font_scale=1.5)

# %%
sigma_vals = np.linspace(0.1, 3, 50)
confidence_vals = np.linspace(0.5, 1.0, 50)

Xplot = sigma_vals
Yplot = confidence_vals
Zplot = np.zeros((len(sigma_vals), len(confidence_vals)))

dis_data = []

it = list(itertools.product(enumerate(sigma_vals), enumerate(confidence_vals)))
for (i, sigma), (j, conf) in tqdm.tqdm(it):
    score = np.log(conf / (1 - conf))
    p = norm.cdf(score / sigma)
    dis = 4 * p * (1 - p)
    Zplot[j, i] = dis
    dis_data.append(dict(sigma=sigma, conf=conf, dis=dis))

# %%
fig, ax = plt.subplots()
CS = plt.contourf(
    Xplot, Yplot, Zplot,
    np.linspace(0, 1, 15),
    cmap="rocket_r")

plt.ylabel("Non-private prediction confidence")
plt.xlabel("Noise scale σ")
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("Disagreement")
cbar.set_ticks([0., 0.25, 0.5, 0.75, 1.0])

plt.savefig("../images/out_pert_dis.pdf", bbox_inches="tight")

# %%
np.random.seed(42)

# number of points per group
n0 = 10_000
n1 = 10_000

# generate data for 0s
mu0 = np.array([1, 1])
var0 = np.array([[1,.5],[.5,1]])

X0 = np.random.multivariate_normal(mu0,var0,size=n0)
y0 = np.zeros(n0)

# generate data for 1s
mu1 = np.array([-1, -1])
var1 = np.array([[1,.1],[.1,1]])

X1 = np.random.multivariate_normal(mu1,var1,size=n1)
y1 = np.ones(n1)
X_outlier = np.array([[-3, 2]])
y_outlier = [1]

# build dataset
X = np.concatenate([X_outlier, X0, X1])
y = np.concatenate([y_outlier, y0, y1])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, stratify=y, shuffle=True)
X_train0 = X_train[y_train == 0][:100]
X_train1 = X_train[y_train == 1][:100]


penalty=0.1
nonpriv_model = lr(penalty="l2", C=penalty, fit_intercept=False).fit(X_train, y_train)
opt_model = lr().fit(X, y)
opt_boundary_x_vals = np.array([-5, 5])
opt_boundary_y_vals = (opt_model.intercept_ - opt_boundary_x_vals * opt_model.coef_[0][0]) \
                      / opt_model.coef_[0][1]

plt.plot(opt_boundary_x_vals, opt_boundary_y_vals)
plt.scatter(X_train0[:,0], X_train0[:,1])
plt.scatter(X_train1[:,0], X_train1[:,1])

# %%
data_norm = np.ceil(np.linalg.norm(X_train, axis=1).max())
data_norm

# %%
models = []
model_metadata = []
eps_vals = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
num_models = 5000
penalty = 0.1
it = list(itertools.product(eps_vals, range(num_models)))

for i, (eps, rep) in enumerate(tqdm.tqdm(it)):
    sens = 2 / (len(X_train) * penalty)
    delta = 1 / (2 * len(X_train))
    sigma = sens * np.sqrt(2 * np.log(1.25 / delta)) / eps
    noise_vec = np.random.RandomState(int(rep + eps)).normal(
        0, sigma,
        size=2,
    )
    model = lr(penalty="l1", C=penalty, fit_intercept=False)
    model.coef_ = np.array([nonpriv_model.coef_[0] + noise_vec])
    model.intercept_ = 0
    model.classes_ = nonpriv_model.classes_
    
    test_acc = model.score(X_test, y_test)
    models.append(model)
    model_metadata.append(dict(
        model_index=i,
        test_acc=test_acc,
        sigma=sigma,
        eps=eps,
        rep=rep,
    ))
    
model_metadata = pd.DataFrame(model_metadata)

# %%
sns.lineplot(data=model_metadata, x="eps", y="test_acc")


# %%
def get_models_by_query(models, q):
    model_indices = list(model_metadata.query(q).model_index)
    result = []
    for model_index in model_indices:
        result.append(models[model_index])
    return result


# %%
it = list(itertools.product(model_metadata.sigma.unique(), range(3)))
est_data = []

for sigma, rep in tqdm.tqdm(it):
    eps = sens * np.sqrt(2 * np.log(1.25 / delta)) / sigma
    query_models = np.array(get_models_by_query(models, f"sigma == {sigma}"))
    index = list(range(len(query_models)))
    np.random.RandomState(rep).shuffle(index)
    shuffled_models = query_models[index]
    eval_data = X_test[:5]
    for n in np.linspace(50, 5000, 50):
        preds_list = []
        n = int(n)
        for model in shuffled_models[range(n)]:
            preds_list.append(model.predict(eval_data))
        pred_mat = np.array(preds_list)
        dis_vals = 4 * pred_mat.var(axis=0)
        dis_data_list = []
        for i, (input_vec, dis_val) in enumerate(zip(eval_data, dis_vals)):
            dis_data_list.append(dict(
                example_id=i,
                x=input_vec[0], y=input_vec[1],
                dis=dis_val,
                eps=eps,
                sigma=sigma,
                rep=rep,
                n=n)
            )
        est_data.append(pd.DataFrame(dis_data_list))

# %%
est_data_df = pd.concat(est_data)


# %%
def compute_theoretical_mult(row):
    sigma = sens * np.sqrt(2 * np.log(1.25 / delta)) / row.eps
    pred_score = nonpriv_model.coef_[0] @ np.array([row.x, row.y])
    c = norm.cdf(pred_score / (np.linalg.norm(np.array([row.x, row.y]), ord=2) * sigma)).squeeze()
    return 4 * c * (1 - c)

est_data_df = est_data_df.assign(theoretical_mult=est_data_df.apply(compute_theoretical_mult, axis=1))


# %%
def compute_theoretical_err(row):
    rho = 0.05
    eta = np.sqrt(np.log(2 / rho) / (2 * row.n))
    return 1 / (row.n - 1) + 4 * row.n / (row.n - 1) * eta * (1 + eta)

est_data_df = est_data_df.assign(theoretical_err=est_data_df.apply(compute_theoretical_err, axis=1))

# %%
est_data_df = est_data_df.assign(err=np.abs(est_data_df.dis - est_data_df.theoretical_mult))

# %%
g = sns.lineplot(
    data=(
        est_data_df
        .query("eps == 0.1")
        .query("example_id == 0")
        .query("n <= 1000")
        .melt(
            id_vars=["n", "eps", "example_id"],
            value_vars=["theoretical_err", "err"]
        )
        .replace(
            dict(
                theoretical_err="Upper bound",
                err="Empirical"
            )
        )
        .rename(
            columns=dict(
                variable="Error type",
                value="Estimation error",
                n="Num. of model samples"
            )
        )
    ),
    x="Num. of model samples",
    y="Estimation error",
    style="Error type",
    style_order=["Empirical", "Upper bound"],
    hue="Error type",
    err_style="bars",
    marker="o",
)

g.get_legend().remove()

g.set_ylabel("Estimation error α")
g.set_xlabel("Number of re-trainings m")

# plt.xscale("log")
plt.yscale("log")
# plt.savefig("../images/estimator_rate_vs_real_error.pdf", bbox_inches="tight")

# %%
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots()
    g = sns.lineplot(est_data_df, x="n", y="theoretical_err")
    g.set_ylabel("Estimation error α")
    g.set_xlabel("Number of re-trainings m")
    g.set_xlim(1000, 5000)
    g.set_ylim(0.07, 0.18)
#     fig.savefig("../images/estimator_rate.pdf", bbox_inches="tight")

# %%
len(get_models_by_query(models, f"eps == {eps_vals[0]} and test_acc >= 0.80"))

# %%
nonpriv_test_acc = nonpriv_model.score(X_test, y_test)
nonpriv_test_acc

# %%
acc_levels = np.linspace(0, 0.01, 10)
acc_level_data = []
it = list(itertools.product(eps_vals, acc_levels))

for eps_val, acc_level in tqdm.tqdm(it):
    query_models = get_models_by_query(
        models,
        f"eps == {eps_val} and test_acc >= {nonpriv_test_acc - acc_level}"
    )
    if len(query_models) == 0:
        continue
        
    preds_list = []
    for model in query_models:
        preds_list.append(model.predict(X_test))
    pred_mat = np.array(preds_list)
    dis_vals = 4 * pred_mat.var(axis=0)
    acc_level_data.append(
        (pd.DataFrame(dict(disagreement=dis_vals))
            .assign(acc_level=acc_level, epsilon=eps_val)
        )
    )

# %%
sns.lineplot(
    data=pd.concat(acc_level_data),
    x="acc_level",
    y="disagreement",
    hue="epsilon",
    marker="o",
    palette="rocket_r",
    err_style="bars",
    legend=False,
)

norm = plt.Normalize(np.min(eps_vals), np.max(eps_vals))
sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm)
cbar.set_label("Privacy param. ε")
plt.ylabel("Disagreement")
plt.xlabel("Accuracy decrease from non-priv.")
