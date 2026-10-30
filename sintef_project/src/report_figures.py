"""Figures for the physics-first model (log 037 / the report). Writes report/figures/physics_*.png.

    python report_figures.py light      # from committed results: walk-forward, ablation, ...
    python report_figures.py model      # needs a model fit: contributions, example week

Palette: the validated three-slot categorical order (blue, orange, aqua) -- blue is always the
LP + LightGBM model, orange the submitted XGBoost model, aqua the raw optimiser. AUCs sit in
0.93-1.0, so comparisons are dot plots (a bar must start at zero; a truncated bar misleads).
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import shoplite as S  # noqa: E402

FIG = S.ROOT / "report/figures"
RUNS = S.ROOT / "report/runs"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 200,
})
FOLD_LABEL = {"2020-01": "2020-H1", "2020-07": "2020-H2", "2021-01": "2021-H1",
              "2021-07": "2021-H2", "2022-01": "2022-H1", "2022-07": "2022-H2"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_thresholds():
    """Implied price threshold per mixed week vs the feeding reservoir's terminal water value."""
    price, inflow, vol, wv, _, _ = S.load_raw()
    uc = pd.read_csv(S.ROOT / "data/kernel/Unit_commitment_decisions.csv", parse_dates=["starttime"])
    P = np.stack([price.reindex(pd.date_range(s, periods=168, freq="h")).to_numpy()
                  for s in uc.starttime])
    panels = [("Byrte_G1", "Botnedalsvatn"), ("Vesle Kjela_G1", "Kjelavatn"),
              ("Kjela_G1", "Foersvatn"), ("Songa_G1", "Songavatn")]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, (u, r) in zip(axes, panels):
        Y = uc[[f"result_committed_{u}_t{h}" for h in range(168)]].to_numpy()
        xs, ys = [], []
        for i, s in enumerate(uc.starttime):
            y = Y[i]
            if 12 <= y.sum() <= 156:
                thr = 0.5 * (np.quantile(P[i][y == 1], 0.1) + np.quantile(P[i][y == 0], 0.9))
                xs.append(wv.loc[s + pd.Timedelta(days=7), r])
                ys.append(thr)
        xs, ys = np.array(xs), np.array(ys)
        b, a = np.polyfit(xs, ys, 1)
        r2 = np.corrcoef(xs, ys)[0, 1] ** 2
        ax.scatter(xs, ys, s=6, color=BLUE, alpha=0.35, linewidths=0)
        lim = [0, np.quantile(np.r_[xs, ys], 0.995)]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(u.replace("_G1", ""))
        ax.text(0.04, 0.96, f"slope {b:.2f}\n$R^2$ {r2:.2f}\nn={len(xs)}", transform=ax.transAxes,
                va="top", fontsize=8, color=INK2)
    axes[0].set_ylabel("implied price threshold (EUR/MWh)")
    fig.supxlabel("terminal water value of the feeding reservoir, start + 7 days (EUR/MWh)",
                  fontsize=9, color=INK2)
    save(fig, "physics_01_threshold_vs_water_value")


def fig_walkforward():
    p = pd.read_csv(RUNS / "walkforward_physics_s3.csv", index_col=0)
    b = pd.read_csv(RUNS / "walkforward_baseline_s3.csv", index_col=0)
    x = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i in (3, 5):                                    # test-like folds
        ax.axvspan(i - 0.4, i + 0.4, color="#f1f0ec", zorder=0, lw=0)
    ax.text(3, 0.9985, "test-like", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.text(5, 0.9985, "test-like", ha="center", va="top", fontsize=7.5, color=INK2)
    series = [("LP + LightGBM", p["micro"], BLUE), ("XGBoost (submitted)", b["micro"], ORANGE),
              ("raw LP, no learning", p["lp_raw"], AQUA)]
    for name, v, c in series:
        ax.plot(x, v.to_numpy(), color=c, marker="o", ms=6, mec=SURF, mew=1.2,
                label=f"{name} (mean {v.mean():.4f})")
    ax.set_xticks(x, [FOLD_LABEL[f] for f in p.index])
    ax.set_xlim(-0.4, len(p) - 0.6)
    ax.set_ylim(0.94, 1.0)
    ax.set_ylabel("micro ROC-AUC")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=3)
    save(fig, "physics_02_walkforward")


def fig_per_unit():
    p = pd.read_csv(RUNS / "walkforward_physics_s3.csv", index_col=0)
    b = pd.read_csv(RUNS / "walkforward_baseline_s3.csv", index_col=0)
    units = [c for c in b.columns if c not in ("n_train_cases", "micro")]
    d = pd.DataFrame({"base": b[units].mean(), "phys": p[units].mean()})
    d = d.sort_values("base")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hlines(y, d["base"], d["phys"], color=GRID, lw=2.5, zorder=1)
    ax.scatter(d["base"], y, color=ORANGE, s=36, zorder=2, label="XGBoost (submitted)",
               edgecolors=SURF, linewidths=1)
    ax.scatter(d["phys"], y, color=BLUE, s=36, zorder=3, label="LP + LightGBM",
               edgecolors=SURF, linewidths=1)
    for yi, (u, r) in zip(y, d.iterrows()):
        ax.text(r["phys"] + 0.004, yi, f"+{r['phys'] - r['base']:.3f}", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(y, [u.replace("_", " ") for u in d.index])
    ax.set_xlim(0.85, 1.02)
    ax.set_xlabel("ROC-AUC, mean over the six walk-forward folds")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "physics_03_per_unit")


def fig_ablation():
    a = pd.read_csv(RUNS / "ablation_physics_s3.csv", index_col=0)
    b = pd.read_csv(RUNS / "walkforward_baseline_s3.csv", index_col=0)["micro"].mean()
    label = {"full": "full model (61 features)", "no_storage_trace": "without observed storage",
             "optimiser_only": "LP features + unit identity only",
             "no_optimiser": "without the LP features"}
    a = a.loc[["full", "no_storage_trace", "optimiser_only", "no_optimiser"]]
    y = np.arange(len(a))[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.axvline(b, color=ORANGE, lw=1.2, ls="--")
    ax.text(b, len(a) - 0.45, f"XGBoost (submitted) {b:.4f}", color=INK2, fontsize=7.5,
            ha="center")
    ax.scatter(a["mean"], y, color=BLUE, s=40, zorder=3, edgecolors=SURF, linewidths=1)
    for yi, v in zip(y, a["mean"]):
        ax.text(v - 0.0015, yi, f"{v:.4f}", va="center", ha="right", fontsize=8, color=INK)
    ax.set_yticks(y, [label[i] for i in a.index])
    ax.set_xlim(0.94, 0.995)
    ax.set_ylim(-0.6, len(a) - 0.2)
    ax.set_xlabel("micro ROC-AUC, mean over six walk-forward folds")
    ax.grid(axis="y", visible=False)
    save(fig, "physics_04_ablation")


def fig_importance():
    s = pd.read_csv(RUNS / "walkforward_physics_s3_importance.csv", index_col=0)["gain_share"]
    s = s.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.barh(np.arange(len(s)), s.to_numpy(), color=BLUE, height=0.7)
    for i, v in enumerate(s.to_numpy()):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(np.arange(len(s)), s.index)
    ax.set_xlabel("share of total split gain (last walk-forward fold)")
    ax.set_xlim(0, s.max() * 1.15)
    ax.grid(axis="y", visible=False)
    save(fig, "physics_05_gain_importance")


def fig_adversarial():
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score

    import adversarial as A
    import shoplite_batch as SB
    d = A.case_table()
    st = SB.all_cases().starttime
    d["period"] = np.where(st.dt.year >= 2023, "test",
                           st.dt.year.astype(str) + np.where(st.dt.month <= 6, "-H1", "-H2"))
    feats = [c for c in d.columns if c not in ("period", "month")]
    test = d[d.period == "test"]
    rows = []
    for per in sorted(p for p in d.period.unique() if p != "test"):
        a = pd.concat([d[d.period == per], test])
        clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                 verbose=-1, n_jobs=1)
        rows.append((per, cross_val_score(clf, a[feats], (a.period == "test").astype(int),
                                          cv=5, scoring="roc_auc").mean()))
    r = pd.DataFrame(rows, columns=["period", "auc"])
    r.to_csv(RUNS / "adversarial_halfyears.csv", index=False)
    y = np.arange(len(r))[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.text(0.505, len(r) - 0.6, "indistinguishable from test", fontsize=7.5, color=INK2)
    ax.scatter(r["auc"], y, color=BLUE, s=30, zorder=3, edgecolors=SURF, linewidths=1)
    for yi, v in zip(y, r["auc"]):
        if v < 0.95:
            ax.text(v - 0.012, yi, f"{v:.3f}", va="center", ha="right", fontsize=7.5)
    ax.set_yticks(y, r["period"])
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("classifier AUC separating the period's weeks from 2023-24 test weeks")
    ax.grid(axis="y", visible=False)
    save(fig, "physics_06_adversarial")


def _fold_model(train_end="2021-07-01", val_end="2022-01-01", stride=3, rounds=600):
    """The 2021-H2 walk-forward fold (most test-like): model, validation rows and features."""
    import physics_model as M
    meta, X, names = M.load()
    st = meta["starttime"].to_numpy()
    y = meta["y"].to_numpy()
    run = meta["run_no"].to_numpy()
    tr = (st < np.datetime64(train_end)) & ~np.isnan(y) & (run % stride == 0)
    va = (st >= np.datetime64(train_end)) & (st < np.datetime64(val_end))
    bst = M.fit(X[tr], y[tr], names, rounds)
    return bst, meta[va].reset_index(drop=True), X[va], names


def fig_model():
    from sklearn.metrics import roc_auc_score
    bst, mv, Xv, names = _fold_model()
    p = bst.predict(Xv)
    print("fold AUC", round(roc_auc_score(mv["y"], p), 4))
    units = [S.sanitize(u) for u in S.load_topology().units]

    # --- contributions (SHAP values from the trees) on a sample of validation rows
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xv), 20_000, replace=False)
    C = bst.predict(Xv[idx], pred_contrib=True)[:, :-1]
    mabs = pd.Series(np.abs(C).mean(0), index=names).sort_values(ascending=False)
    group = mabs.groupby(mabs.index.str.split("_").str[0]).sum().sort_values(ascending=False)
    gname = {"lp": "lp_  optimiser solution", "s": "s_  unit identity, static",
             "ec": "ec_  economics", "cn": "cn_  constraints", "st": "st_  observed storage"}
    pd.DataFrame({"mean_abs_contribution": mabs}).to_csv(RUNS / "contributions_2021H2.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    top = mabs.head(15)[::-1]
    a1.barh(np.arange(len(top)), top.to_numpy(), color=BLUE, height=0.7)
    a1.set_yticks(np.arange(len(top)), top.index)
    a1.set_xlabel("mean |contribution| to the log-odds")
    a1.set_title("Top 15 features", loc="left")
    a1.grid(axis="y", visible=False)
    g = (group / group.sum())[::-1]
    a2.barh(np.arange(len(g)), g.to_numpy(), color=BLUE, height=0.6)
    for i, v in enumerate(g.to_numpy()):
        a2.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8, color=INK2)
    a2.set_yticks(np.arange(len(g)), [gname[k] for k in g.index])
    a2.set_xlim(0, 1)
    a2.set_xlabel("share of total |contribution|")
    a2.set_title("By feature group", loc="left")
    a2.grid(axis="y", visible=False)
    save(fig, "physics_07_contributions")

    # --- how the learned layer reads the LP: SHOP on-rate against LP commitment
    lpu = Xv[:, names.index("lp_u")]
    bins = np.array([-0.01, 0.001, 0.2, 0.4, 0.6, 0.8, 0.999, 1.01])
    mids = ["0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1", "1"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for lab, mask, c in [("all units", np.ones(len(lpu), bool), BLUE),
                         ("Lio G1", mv["unit"].to_numpy() == units.index("Lio_G1"), ORANGE),
                         ("Hogga G1", mv["unit"].to_numpy() == units.index("Hogga_G1"), AQUA)]:
        b = np.digitize(lpu[mask], bins) - 1
        rate = [mv["y"].to_numpy()[mask][b == k].mean() if (b == k).sum() > 200 else np.nan
                for k in range(len(mids))]
        ax.plot(np.arange(len(mids)), rate, color=c, marker="o", ms=6, mec=SURF, mew=1.2, label=lab)
    ax.plot([0, len(mids) - 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xticks(np.arange(len(mids)), mids)
    ax.set_xlabel("LP commitment u (relaxation) in the hour")
    ax.set_ylabel("share of hours SHOP commits")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "physics_08_lp_vs_shop")

    # --- one test-like week: SHOP, the LP, the model
    mv["p"] = p
    mv["lp_u"] = lpu
    wk = mv.groupby("run_no").apply(lambda g: ((g.y > 0.5) != (g.lp_u > 0.5)).mean())
    run = wk.index[np.argsort(wk.to_numpy())[int(0.75 * len(wk))]]   # a week the LP finds hard
    g = mv[mv.run_no == run]
    start = pd.Timestamp(g["starttime"].iloc[0]).date()
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#9ec5f4", "#2a78d6", "#0d366b"])
    for ax, col, title in zip(axes, ["y", "lp_u", "p"],
                              ["SHOP commitment (label)", "LP relaxation u (no learning)",
                               "LP + LightGBM probability"]):
        M = g.pivot(index="unit", columns="hour", values=col).to_numpy()
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(np.arange(len(units)), [u.replace("_", " ") for u in units], fontsize=7)
        ax.set_title(title, loc="left")
        ax.grid(False)
    axes[-1].set_xticks(np.arange(0, 169, 24), [f"day {d}" for d in range(8)])
    fig.colorbar(im, ax=axes, shrink=0.6, label="on (1) / off (0)")
    fig.suptitle(f"Case starting {start} (2021-H2 fold): AUC LP {roc_auc_score(g.y, g.lp_u):.3f}, "
                 f"model {roc_auc_score(g.y, g.p):.3f}", fontsize=9, color=INK2, x=0.42)
    fig.savefig(FIG / "physics_09_example_week.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote physics_09_example_week", start)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "light"
    if what == "light":
        fig_walkforward(); fig_per_unit(); fig_ablation(); fig_importance()
        fig_thresholds(); fig_adversarial()
    elif what == "model":
        fig_model()
    elif what in globals():
        globals()[what]()
