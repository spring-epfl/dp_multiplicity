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
from matplotlib import animation

sns.set(style="white", context="paper", font_scale=1.5)

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
nonpriv_model = lr().fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, stratify=y)
X_train0 = X_train[y_train == 0][:100]
X_train1 = X_train[y_train == 1][:100]

opt_boundary_x_vals = np.array([-5, 5])
opt_boundary_y_vals = (nonpriv_model.intercept_ - opt_boundary_x_vals * nonpriv_model.coef_[0][0]) \
                      / nonpriv_model.coef_[0][1]
plt.plot(opt_boundary_x_vals, opt_boundary_y_vals)
plt.scatter(X_train0[:,0], X_train0[:,1])
plt.scatter(X_train1[:,0], X_train1[:,1])

# %%
data_norm = np.ceil(np.linalg.norm(X_train, axis=1).max())
data_norm

# %%
models = []
model_metadata = []
# eps_vals = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
eps_vals = [0.1, 0.5, 1.0, 1.5]
num_models = 5000
it = list(itertools.product(eps_vals, range(num_models)))

for i, (eps, rep) in enumerate(tqdm.tqdm(it)):
    model = diffprivlib.models.LogisticRegression(data_norm=data_norm, epsilon=eps)
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    models.append(model)
    model_metadata.append(dict(
        model_index=i,
        test_acc=test_acc,
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
len(get_models_by_query(models, f"eps == {eps_vals[0]} and test_acc >= 0.80"))

# %%
nonpriv_test_acc = nonpriv_model.score(X_test, y_test)

# %%
acc_levels = np.linspace(0, 0.01, 10)
acc_level_data = []
it = list(itertools.product(eps_vals, acc_levels))

for eps_val, acc_level in tqdm.tqdm(it):
    query_models = get_models_by_query(
        models,
        f"eps == {eps_val} and test_acc >= {nonpriv_test_acc - acc_level}"
    )
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
    estimator=lambda s: np.quantile(s, .95),
    marker="o",
    errorbar=None,
    err_style="bars",
    palette="rocket_r",
    legend=False,
)

norm = plt.Normalize(np.min(eps_vals), np.max(eps_vals))
sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm)
cbar.set_label("Privacy param. ε")
plt.ylabel("Disagreement (top 5%)")
plt.xlabel("Accuracy decrease from optimal")

# plt.savefig('../images/mult_vs_acc.pdf')

# %%
def prepare_mult_map(models, grid_resolution=50, x_range=(-5, 5), y_range=(-5, 5)):
    Xplot = np.linspace(*x_range, grid_resolution)
    Yplot = np.linspace(*y_range, grid_resolution)
    Zplot = np.zeros([grid_resolution, grid_resolution])
    
    it = list(itertools.product(enumerate(Xplot), enumerate(Yplot)))
    for ((i, x), (j, y)) in tqdm.tqdm(it):
        scores = []
        for model in models:
            scores.append(model.predict(np.array([[x, y]])))
        scores = np.array(scores)
        Zplot[i, j] = scores.var()
    
    return Xplot, Yplot, Zplot


# %%
def get_mult_map_data(models, eps, test_acc=None, **kwargs):
    if test_acc is not None:
        test_acc = 0.5
    query_models = get_models_by_query(models, f"eps == {eps} and test_acc >= {test_acc}")
    Xplot, Yplot, Zplot = prepare_mult_map(query_models)
    return Xplot, Yplot, Zplot


# %%
plot_eps_vals = [0.1, 0.5, 1.0, 1.5]

# %%
plot_data_by_eps = {}
for eps in plot_eps_vals:
    plot_data_by_eps[eps] = get_mult_map_data(models, eps, test_acc=0.0)

# %%
fig, axes = plt.subplots(
    nrows=1,
    ncols=4,
    sharey=True,
    constrained_layout=True,
    figsize=(12, 3)
)

num_points = 75

def plot_eps(ax, eps_val, Xplot, Yplot, Zplot):
    ax.set_title(f"ε = {eps_val:1.1f}")
    CS = ax.contourf(Xplot, Yplot, 4 * Zplot,
                     np.linspace(0, 1, 15),
                     cmap="rocket_r")
    ax.scatter(X_train0[:num_points,0], X_train0[:num_points,1],
               edgecolor="white",
               c="crimson", marker="P", s=55)
    ax.scatter(X_train1[:num_points,0], X_train1[:num_points,1],
               edgecolor="white",
               c="grey", marker="X", s=50)
    ax.plot(opt_boundary_x_vals, opt_boundary_y_vals,
            color="white",
            alpha=0.75,
            zorder=999999,
            linestyle="--")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    return CS
   
for i, (eps_val, datasets) in enumerate(reversed(plot_data_by_eps.items())):
    CS = plot_eps(axes[i], eps_val, *datasets)

cbar = fig.colorbar(CS)
cbar.set_ticks([0., 0.25, 0.5, 0.75, 1.0])
cbar.ax.set_ylabel("Disagreement")
plt.savefig('../images/multiplicity_maps.pdf')

# %%
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
# cs = ax.contourf(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), 200)

norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
sm.set_array([])

ax.set_title("Decision variance. Privacy leakage ε = ")

class MultAnimation:
    def __init__(self):
        self.cb = None

    def animate(self, i):
        ax.clear()
        if self.cb is not None:
            if hasattr(self.cb, "clear"):
                self.cb.clear()
        eps = plot_eps_vals[i]
        ax.set_title(f"Decision variance. Privacy leakage ε = {eps:.2f}")
        Xplot, Yplot, Zplot = plot_data_by_eps[eps]
    #     levels = np.linspace(0, 1.0, 200)
        ax.contourf(Xplot, Yplot, 4 * Zplot,
                    np.linspace(0, 1, 100),
                    cmap="rocket_r")
        self.cb = fig.colorbar(sm)
        ax.scatter(X_train0[:,0], X_train0[:,1],
                   edgecolor="white",
                   c="crimson", marker="P", s=55)
        ax.scatter(X_train1[:,0], X_train1[:,1],
                   edgecolor="white",
                   c="grey", marker="X", s=50)
        ax.plot(opt_boundary_x_vals, opt_boundary_y_vals,
                color="white",
                alpha=0.75,
                zorder=999999,
                linestyle="--")
    
animator = MultAnimation()

anim = animation.FuncAnimation(fig, animator.animate, range(len(plot_data_by_eps)), interval=500, blit=False)
anim.save("../images/mult_anim.gif")
