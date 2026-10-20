"""Пространство поиска RAG-пайплайна: retriever -> reranker -> filter."""
import itertools

EMBED_MODELS = {
    "rubert-tiny2": dict(name="cointegrated/rubert-tiny2", q_prefix="", d_prefix=""),
    "me5-small": dict(name="intfloat/multilingual-e5-small", q_prefix="query: ", d_prefix="passage: "),
    "mminilm": dict(name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    q_prefix="", d_prefix=""),
}
RERANKERS = {
    "mminilm-ce": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "rubert-ce": "DiTy/cross-encoder-russian-msmarco",
    "bge-m3-rr": "BAAI/bge-reranker-v2-m3",
}
RETRIEVERS = ("bm25", "dense", "hybrid")
MIN_TOP_K, MAX_TOP_K, MAX_RERANK_K = 5, 30, 10
STAGES = ("retriever", "reranker", "filter")

DEFAULTS = dict(retriever="bm25", embed=None, k1=1.2, b=0.75, alpha=None, top_k=10,
                reranker="none", rerank_top_k=None, filter="none", filter_thr=None)


def _rnd(x, step):
    return round(round(float(x) / step) * step, 4)


def normalize(cfg):
    """
    Заполняет дефолты, обнуляет неактивные параметры, округляет непрерывные (для кэша)
    и применяет ограничение rerank_top_k <= top_k (как в статье, Appendix D).
    """
    c = {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
    r = c["retriever"]
    c["embed"] = (c["embed"] or "rubert-tiny2") if r in ("dense", "hybrid") else None
    if r in ("bm25", "hybrid"):
        c["k1"], c["b"] = _rnd(c["k1"], 0.1), _rnd(c["b"], 0.05)
    else:
        c["k1"] = c["b"] = None
    c["alpha"] = _rnd(c["alpha"] if c["alpha"] is not None else 0.5, 0.05) if r == "hybrid" else None
    c["top_k"] = int(c["top_k"])
    if c["reranker"] == "none":
        c["rerank_top_k"] = None
    else:
        c["rerank_top_k"] = min(int(c["rerank_top_k"] or 5), c["top_k"])
    c["filter_thr"] = None if c["filter"] == "none" else _rnd(c["filter_thr"] if c["filter_thr"] is not None else 0.5, 0.05)
    return c


def cfg_key(cfg):
    return tuple(sorted(normalize(cfg).items()))


# Optuna (define-by-run, условные параметры)
def optuna_suggest(trial, stages):
    p = {}
    if "retriever" in stages:
        r = p["retriever"] = trial.suggest_categorical("retriever", RETRIEVERS)
        if r in ("dense", "hybrid"):
            p["embed"] = trial.suggest_categorical("embed", list(EMBED_MODELS))
        if r in ("bm25", "hybrid"):
            p["k1"] = trial.suggest_float("k1", 0.5, 2.0)
            p["b"] = trial.suggest_float("b", 0.3, 1.0)
        if r == "hybrid":
            p["alpha"] = trial.suggest_float("alpha", 0.1, 0.9)
        p["top_k"] = trial.suggest_int("top_k", MIN_TOP_K, MAX_TOP_K)
    if "reranker" in stages:
        rr = p["reranker"] = trial.suggest_categorical("reranker", ["none", *RERANKERS])
        if rr != "none":
            p["rerank_top_k"] = trial.suggest_int("rerank_top_k", 1, MAX_RERANK_K)
    if "filter" in stages:
        f = p["filter"] = trial.suggest_categorical("filter", ["none", "relative"])
        if f == "relative":
            p["filter_thr"] = trial.suggest_float("filter_thr", 0.05, 0.95)
    return p


# Сетка для grid search (покомпонентно, как в AutoRAG)
def grid_configs(stage):
    if stage == "retriever":
        base = ([dict(retriever="bm25", k1=k1, b=b) for k1, b in itertools.product((0.9, 1.2, 1.5), (0.5, 0.75, 0.9))]
                + [dict(retriever="dense", embed=e) for e in EMBED_MODELS]
                + [dict(retriever="hybrid", embed=e, alpha=a) for e in EMBED_MODELS for a in (0.3, 0.5, 0.7)])
        return [{**c, "top_k": k} for c in base for k in (5, 10, 20, 30)]
    if stage == "reranker":
        return [dict(reranker="none")] + [dict(reranker=m, rerank_top_k=k) for m in RERANKERS for k in (1, 3, 5, 10)]
    if stage == "filter":
        return [dict(filter="none")] + [dict(filter="relative", filter_thr=t) for t in (0.3, 0.5, 0.7, 0.9)]
    raise ValueError(stage)
