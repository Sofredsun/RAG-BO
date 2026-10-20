import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def pareto(df):
    """Конфигурации, которые не доминируются ни одной другой по (F1 ↑, латентность ↓)."""
    front, best = [], -1.0
    for _, r in df.sort_values(["cost_ms", "f1"], ascending=[True, False]).iterrows():
        if r.f1 > best:
            front.append(r); best = r.f1
    return pd.DataFrame(front)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rus-scifact")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    t = pd.read_csv(out / f"trials_{a.data}.csv")
    f = pd.read_csv(out / f"final_{a.data}.csv")
    t["run"] = t.scope + " / " + t.method
    g_ = t.groupby(["run", "seed"])
    t["best"] = g_.f1.cummax()
    t["sim_min"] = g_.sim_s.cumsum() / 60

    # Сходимость: по числу trials (среднее ± std по сидам) и по оценке времени без кэша
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (run, g) in enumerate(t.groupby("run")):
        color, ls = f"C{i}", ("--" if run.startswith("local") else "-")
        piv = g.pivot_table(index="trial", columns="seed", values="best")
        m, s = piv.mean(axis=1), piv.std(axis=1).fillna(0)
        ax[0].plot(m.index, m, ls, color=color, label=run)
        ax[0].fill_between(m.index, m - s, m + s, color=color, alpha=.12)
        for j, (_, gs) in enumerate(g.groupby("seed")):
            ax[1].step(gs.sim_min, gs.best, ls, where="post", color=color, alpha=.7, label=run if j == 0 else None)
    ax[0].set(xlabel="номер trial", ylabel="лучший F1 (валидация)", title="Сходимость по числу оценок")
    ax[1].set(xlabel="время оптимизации без кэша (оценка), мин", ylabel="лучший F1 (валидация)",
              title="Сходимость по стоимости")
    for x in ax:
        x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"convergence_{a.data}.png", dpi=150)

    # Парето-фронт F1 vs латентность для глобальных запусков (первый сид)
    runs = sorted(r for r in t.run.unique() if r.startswith("global"))
    if runs:
        fig, ax = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 4), squeeze=False)
        for x, run in zip(ax[0], runs):
            g = t[t.run == run]; g = g[g.seed == g.seed.min()]
            p = pareto(g)
            x.scatter(g.f1, g.cost_ms, s=12, alpha=.4, label="все trials")
            x.scatter(p.f1, p.cost_ms, marker="x", color="red", label="Парето-фронт")
            x.set(title=run, xlabel="F1", ylabel="латентность, мс/запрос"); x.grid(alpha=.3); x.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"pareto_{a.data}.png", dpi=150)

    # Сводная таблица (val — запросы, на которых оптимизировали; test — отложенные)
    agg = f.groupby(["scope", "method"]).agg(
        runs=("seed", "count"), trials=("n_trials", "mean"),
        val_f1=("val_f1", "mean"), val_std=("val_f1", "std"),
        test_f1=("test_f1", "mean"), test_std=("test_f1", "std"),
        latency_ms=("test_cost_ms", "mean"), opt_time_min=("sim_time_s", lambda s: s.mean() / 60)).round(4)
    agg.to_csv(out / f"summary_{a.data}.csv")
    print(agg.to_string())


if __name__ == "__main__":
    main()
