"""
Individual vs group fairness on synthetic two-group social networks.

Question: how do network structure (group size, homophily) and the choice of
similarity metric affect (a) InFoRM-style individual bias of graph-mining
outputs and (b) simple group-fairness measures?

Pipeline per (minority share, homophily, seed):
  1. Sample a two-group stochastic block model (networkx).
  2. Run PageRank and spectral clustering (Ng-Jordan-Weiss).
  3. Compute individual bias tr(Y^T L_S Y) / tr(Y^T D_S Y) under three
     node-similarity matrices S: Jaccard (structural), feature-based
     "informative" (features carry group signal), feature-based "random".
  4. Compute group measures: minority exposure ratio, top-10% representation
     ratio, NMI between clusters and groups.

Usage:  python experiment.py [--seeds 20] [--n 400] [--out results]
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

P_IN = 0.10
HOMOPHILY = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]  # p_out / p_in; 1.0 = no homophily
MINORITY_SHARES = [0.50, 0.30, 0.15]
SIMS = ["jaccard", "feat_informative", "feat_random"]
TASKS = ["pagerank", "spectral"]


# ----------------------------------------------------------------- graphs
def sample_sbm(n, minority_share, h, seed):
    n_min = int(round(minority_share * n))
    sizes = [n - n_min, n_min]  # block 0 = majority, block 1 = minority
    p = [[P_IN, h * P_IN], [h * P_IN, P_IN]]
    G = nx.stochastic_block_model(sizes, p, seed=seed)
    A = nx.to_numpy_array(G, nodelist=range(n))
    groups = np.array([0] * sizes[0] + [1] * sizes[1])
    return G, A, groups


# ------------------------------------------------------- mining algorithms
def pagerank(A, alpha=0.85, tol=1e-12, max_iter=1000):
    n = A.shape[0]
    deg = A.sum(1)
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        contrib = np.where(deg > 0, r / np.maximum(deg, 1e-300), 0.0)
        dangling = r[deg == 0].sum()
        r_new = alpha * (A.T @ contrib + dangling / n) + (1 - alpha) / n
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return r / r.sum()


def spectral_embedding(A, k=2):
    """NJW: top-k eigenvectors of D^-1/2 A D^-1/2, rows normalised to unit length."""
    deg = A.sum(1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    N = dinv[:, None] * A * dinv[None, :]
    vals, vecs = np.linalg.eigh(N)
    X = vecs[:, -k:]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def cluster_labels(Y, k, rng):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, lab = kmeans2(Y, k, minit="++", seed=rng)
    return lab


# -------------------------------------------------------------- similarity
def jaccard_similarity(A):
    inter = A @ A
    deg = A.sum(1)
    union = deg[:, None] + deg[None, :] - inter
    S = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(S, 0.0)
    return S


def knn_rbf_similarity(F, k=10):
    """Symmetrised k-nearest-neighbour graph with RBF weights on feature distance."""
    d2 = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    idx = np.argsort(d2, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(F)), k)
    cols = idx.ravel()
    sigma2 = np.median(d2[rows, cols])
    S = np.zeros_like(d2)
    S[rows, cols] = np.exp(-d2[rows, cols] / sigma2)
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, 0.0)
    return S


def make_features(groups, rng, dim=5, shift=1.5):
    """informative: group-dependent mean shift + noise; random: pure noise."""
    noise = rng.normal(size=(len(groups), dim))
    informative = noise + shift * groups[:, None]
    random_ = rng.normal(size=(len(groups), dim))
    return informative, random_


# ----------------------------------------------------------------- metrics
def individual_bias(Y, S):
    """Relative InFoRM bias: tr(Y^T L_S Y) / tr(Y^T D_S Y), with L_S = D_S - S."""
    Y = Y.reshape(len(Y), -1)
    d = S.sum(1)
    num = np.trace(Y.T @ (np.diag(d) - S) @ Y)
    den = (d[:, None] * Y * Y).sum()
    return num / den


def nmi(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ca, cb = np.unique(a), np.unique(b)
    cont = np.array([[np.sum((a == i) & (b == j)) for j in cb] for i in ca], float)
    pxy = cont / cont.sum()
    px, py = pxy.sum(1), pxy.sum(0)
    nz = pxy > 0
    mi = (pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return 0.0 if hx == 0 or hy == 0 else mi / np.sqrt(hx * hy)


def exposure_ratio(pr, groups):
    """mean PageRank of minority / mean PageRank of majority (1 = parity)."""
    return pr[groups == 1].mean() / pr[groups == 0].mean()


def top_rep_ratio(pr, groups, frac=0.10):
    """share of minority among top-frac nodes / minority share overall (1 = parity)."""
    k = max(1, int(round(frac * len(pr))))
    top = np.argsort(-pr)[:k]
    return groups[top].mean() / groups.mean()


# ------------------------------------------------------------ self-checks
def self_checks(seed=0):
    G, A, groups = sample_sbm(200, 0.3, 0.3, seed)
    rng = np.random.default_rng(seed)
    # 1) PageRank matches networkx
    ours = pagerank(A)
    ref = np.array([nx.pagerank(G, alpha=0.85, tol=1e-12)[i] for i in range(len(A))])
    assert np.abs(ours - ref).max() < 1e-8, "PageRank differs from networkx"
    # 2) trace identity: tr(Y^T L_S Y) == 1/2 sum_ij S_ij ||y_i - y_j||^2
    S = jaccard_similarity(A)
    Y = spectral_embedding(A)
    L = np.diag(S.sum(1)) - S
    lhs = np.trace(Y.T @ L @ Y)
    rhs = 0.5 * (S * ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)).sum()
    assert abs(lhs - rhs) < 1e-8 * max(1, abs(lhs)), "trace identity failed"
    # 3) NMI sanity
    assert abs(nmi(groups, groups) - 1) < 1e-9
    print(f"self-checks passed (trace identity: {lhs:.6f} == {rhs:.6f})")


# -------------------------------------------------------------- experiment
def run_one(n, share, h, seed):
    rng = np.random.default_rng(seed)
    G, A, groups = sample_sbm(n, share, h, seed)
    Fi, Fr = make_features(groups, rng)
    S = {
        "jaccard": jaccard_similarity(A),
        "feat_informative": knn_rbf_similarity(Fi),
        "feat_random": knn_rbf_similarity(Fr),
    }
    pr = pagerank(A)
    emb = spectral_embedding(A)
    lab = cluster_labels(emb, 2, rng)
    out = dict(
        share=share,
        homophily=h,
        seed=seed,
        exposure_ratio=exposure_ratio(pr, groups),
        top10_rep_ratio=top_rep_ratio(pr, groups),
        nmi=nmi(lab, groups),
    )
    for sname, Smat in S.items():
        out[f"bias_pagerank_{sname}"] = individual_bias(pr, Smat)
        out[f"bias_spectral_{sname}"] = individual_bias(emb, Smat)
    return out


# ---------------------------------------------------------------- plotting
# Categorical slots 1-3 of the reference palette (validated all-pairs), light surface.
COLORS = {0.50: "#2a78d6", 0.30: "#eb6834", 0.15: "#1baf7a"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def style_ax(ax, title, ylabel, xlabel=True):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    if xlabel:
        ax.set_xlabel("homophily  (p_out / p_in, lower = more homophilous)", fontsize=8, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_xlim(1.04, 0.0)  # left = no homophily, right = strong homophily


def plot_lines(ax, summ, col, ref=None):
    for share in MINORITY_SHARES:
        d = summ[summ.share == share].sort_values("homophily")
        ax.plot(d.homophily, d[(col, "mean")], color=COLORS[share], lw=2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"minority {int(share*100)}%")
        ax.fill_between(d.homophily, d[(col, "mean")] - d[(col, "std")],
                        d[(col, "mean")] + d[(col, "std")], color=COLORS[share], alpha=0.12, lw=0)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)))


def make_figures(summ, outdir):
    # Figure 1: group measures
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=SURFACE)
    plot_lines(axes[0], summ, "exposure_ratio", ref=1.0)
    style_ax(axes[0], "Minority PageRank exposure", "minority / majority mean score  (1 = parity)")
    plot_lines(axes[1], summ, "top10_rep_ratio", ref=1.0)
    style_ax(axes[1], "Minority share of PageRank top 10%", "share in top 10% / overall share  (1 = parity)")
    plot_lines(axes[2], summ, "nmi")
    style_ax(axes[2], "Spectral clusters vs groups", "NMI  (1 = clusters recover groups)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Group-fairness measures on two-group stochastic block models (mean ± 1 sd over seeds)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / "fig1_group_measures.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: individual bias, tasks x similarity metrics
    names = {"jaccard": "Jaccard (structural)", "feat_informative": "Features: informative",
             "feat_random": "Features: random"}
    task_names = {"pagerank": "PageRank", "spectral": "Spectral embedding"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), facecolor=SURFACE, sharex=True)
    for r, task in enumerate(TASKS):
        for c, sname in enumerate(SIMS):
            ax = axes[r, c]
            plot_lines(ax, summ, f"bias_{task}_{sname}")
            style_ax(ax, f"{task_names[task]} · {names[sname]}",
                     "relative individual bias" if c == 0 else "", xlabel=(r == 1))
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Individual bias tr(YᵀL_S Y)/tr(YᵀD_S Y): the same output looks different under different similarity metrics",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "fig2_individual_bias.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    self_checks()

    rows = []
    for share in MINORITY_SHARES:
        for h in HOMOPHILY:
            for seed in range(args.seeds):
                rows.append(run_one(args.n, share, h, seed))
        print(f"done minority share {share:.2f}")
    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "raw_results.csv", index=False)

    metric_cols = [c for c in raw.columns if c not in ("share", "homophily", "seed")]
    summ = raw.groupby(["share", "homophily"])[metric_cols].agg(["mean", "std"]).reset_index()
    summ.columns = [a if not b else (a, b) for a, b in summ.columns]
    flat = summ.copy()
    flat.columns = [a if isinstance(a, str) else f"{a[0]}_{a[1]}" for a in flat.columns]
    flat.to_csv(outdir / "summary.csv", index=False)

    make_figures(summ, outdir)
    print(f"wrote results to {outdir}/")


if __name__ == "__main__":
    main()
