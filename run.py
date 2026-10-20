import argparse, csv, json, random, time, warnings
from pathlib import Path

import optuna

from pipeline import RAGEvaluator, STBackend, ToyBackend, load_rus_scifact, make_toy
from space import STAGES, grid_configs, normalize, optuna_suggest

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

FIELDS = ["data", "method", "scope", "seed", "trial", "stage", "f1", "precision", "recall", "cost_ms", "sim_s", "wall_s", "cfg"]


def run_stage(ev, method, stages, fixed, budget, seed, log):
    """
    Оптимизирует параметры указанных стадий при зафиксированных остальных.
    Возвращает лучший полный конфиг.
    """
    best = {"f1": -1.0, "cfg": None}

    def objective(partial):
        cfg = normalize({**fixed, **partial})
        t0 = time.perf_counter()
        m = ev.evaluate(cfg)
        log(stages, cfg, m, time.perf_counter() - t0)
        if m["f1"] > best["f1"]:
            best.update(f1=m["f1"], cfg=cfg)
        return m["f1"]

    if method == "grid":
        assert len(stages) == 1, "grid search поддерживается только покомпонентно (как в AutoRAG)"
        for p in grid_configs(stages[0]):
            objective(p)
    elif method in ("random", "tpe"):
        sampler = (optuna.samplers.RandomSampler(seed=seed) if method == "random" else
                   optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(3, budget // 4),
                                              multivariate=True, group=True))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda t: objective(optuna_suggest(t, stages)), n_trials=budget)
    else:
        raise ValueError(method)
    return best["cfg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rus-scifact", choices=["rus-scifact", "toy"])
    ap.add_argument("--methods", nargs="+", default=["grid", "random", "tpe"], choices=["grid", "random", "tpe"])
    ap.add_argument("--scopes", nargs="+", default=["local", "global"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=60, help="всего trials; в local делится поровну на 3 стадии")
    ap.add_argument("--n-opt-queries", type=int, default=150)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    corpus, queries, qr_train, qr_test = load_rus_scifact() if a.data == "rus-scifact" else make_toy()
    backend = STBackend(a.device) if a.data == "rus-scifact" else ToyBackend(list(corpus.values()))
    opt_q = sorted(random.Random(0).sample(sorted(qr_train), min(a.n_opt_queries, len(qr_train))))
    ev = RAGEvaluator(corpus, queries, qr_train, opt_q, backend, cache_dir=f"cache/{a.data}")
    ev_test = RAGEvaluator(corpus, queries, qr_test, sorted(qr_test), backend, cache_dir=f"cache/{a.data}")
    print(f"Корпус: {len(corpus)} док., запросов: оптимизация {len(ev.qids)}, тест {len(ev_test.qids)}")

    out = Path(a.out); out.mkdir(exist_ok=True)
    tpath, fpath = out / f"trials_{a.data}.csv", out / f"final_{a.data}.csv"
    new_t, new_f = not tpath.exists(), not fpath.exists()
    tf, ff = open(tpath, "a", newline=""), open(fpath, "a", newline="")
    tw = csv.DictWriter(tf, FIELDS)
    fw = csv.DictWriter(ff, ["data", "method", "scope", "seed", "n_trials", "val_f1", "test_f1", "test_recall",
                             "test_cost_ms", "sim_time_s", "wall_s", "cfg"])
    if new_t: tw.writeheader()
    if new_f: fw.writeheader()

    for scope in a.scopes:
        for method in a.methods:
            if method == "grid" and scope == "global":
                print("  пропуск: global grid (пространство слишком велико — как и в статье)")
                continue
            for seed in (a.seeds[:1] if method == "grid" else a.seeds):
                st = {"n": 0, "sim": 0.0, "wall": 0.0}

                def log(stages, cfg, m, wall):
                    sim = m["cost_ms"] * len(ev.qids) / 1000  # сколько секунд занял бы trial без кэша
                    st["n"] += 1; st["sim"] += sim; st["wall"] += wall
                    tw.writerow(dict(data=a.data, method=method, scope=scope, seed=seed, trial=st["n"],
                                     stage="+".join(stages), **{k: round(v, 5) for k, v in m.items()},
                                     sim_s=round(sim, 3), wall_s=round(wall, 4), cfg=json.dumps(cfg)))

                t0 = time.time()
                if scope == "local":
                    cfg = {}
                    for stage in STAGES:
                        cfg = run_stage(ev, method, [stage], cfg, a.budget // len(STAGES), seed, log)
                else:
                    cfg = run_stage(ev, method, list(STAGES), {}, a.budget, seed, log)
                val, test = ev.evaluate(cfg), ev_test.evaluate(cfg)
                fw.writerow(dict(data=a.data, method=method, scope=scope, seed=seed, n_trials=st["n"],
                                 val_f1=round(val["f1"], 4), test_f1=round(test["f1"], 4),
                                 test_recall=round(test["recall"], 4), test_cost_ms=round(test["cost_ms"], 2),
                                 sim_time_s=round(st["sim"], 1), wall_s=round(time.time() - t0, 1),
                                 cfg=json.dumps(cfg)))
                tf.flush(); ff.flush(); ev.save(); ev_test.save()
                print(f"{scope:6s} {method:6s} seed={seed}  trials={st['n']:3d}  val F1={val['f1']:.4f}  "
                      f"test F1={test['f1']:.4f}  sim_time={st['sim']:.0f}s  wall={time.time() - t0:.1f}s")
    tf.close(); ff.close()


if __name__ == "__main__":
    main()
