import json, pickle, random, re, time, urllib.request, zlib
from functools import lru_cache
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from space import EMBED_MODELS, RERANKERS, normalize, cfg_key

HF_DATA = "https://huggingface.co/datasets/kaengreg/{repo}/resolve/main/{file}"
RUS_SCIFACT_FILES = [("rus-scifact", "corpus.jsonl"), ("rus-scifact", "queries.jsonl"),
                     ("rus-scifact-qrels", "train.tsv"), ("rus-scifact-qrels", "test.tsv")]

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass


# Данные
def _download(repo, file, dest):
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=f"kaengreg/{repo}", filename=file, repo_type="dataset", local_dir=dest.parent)
    except ImportError:
        urllib.request.urlretrieve(HF_DATA.format(repo=repo, file=file), dest)


def load_rus_scifact(data_dir="data"):
    """rus-SciFact из бенчмарка RusBEIR (Kovalev et al., 2025): перевод SciFact на русский, формат BEIR."""
    root = Path(data_dir) / "rus-scifact"
    root.mkdir(parents=True, exist_ok=True)
    for repo, file in RUS_SCIFACT_FILES:
        dest = root / file
        if not dest.exists():
            print(f"Скачиваю {repo}/{file}...")
            try:
                _download(repo, file, dest)
            except Exception as e:
                dest.unlink(missing_ok=True)
                urls = "\n".join(HF_DATA.format(repo=r, file=f) for r, f in RUS_SCIFACT_FILES)
                raise SystemExit(f"Не удалось скачать rus-SciFact: {e}\n"
                                 f"Скачайте вручную 4 файла и положите их в {root.resolve()}:\n{urls}")

    def jsonl(p):
        with open(p, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    corpus = {str(d["_id"]): f'{d.get("title") or ""}. {d.get("text") or ""}'.strip(". ")
              for d in jsonl(root / "corpus.jsonl")}
    queries = {str(q["_id"]): q["text"] for q in jsonl(root / "queries.jsonl")}

    def qrels(split):
        out = {}
        with open(root / f"{split}.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3 and int(float(parts[2])) > 0:
                    out.setdefault(parts[0], set()).add(parts[1])
        return out

    return corpus, queries, qrels("train"), qrels("test")


# BM25 для русского
RU_STOPWORDS = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было
вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам
ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз
тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее
сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про
всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более
всегда конечно всю между это также при который которые которая которых является""".split())
_TOKEN = re.compile(r"\w+", re.UNICODE)


_STEMMER = None


@lru_cache(maxsize=None)
def _stem(word):
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem.snowball import SnowballStemmer
        _STEMMER = SnowballStemmer("russian")
    return _STEMMER.stem(word)


def ru_analyzer(text):
    """
    Токены для BM25: нижний регистр, без стоп-слов,
    стемминг Snowball (щитовидной/щитовидная -> щитовидн).
    """
    return [_stem(w) for w in _TOKEN.findall(text.lower().replace("ё", "е")) if len(w) > 1 and w not in RU_STOPWORDS]


def make_toy(seed=0, n_topics=60, docs_per_topic=20, n_queries=500):
    """Синтетический корпус только для проверки кода без скачивания моделей."""
    rng = random.Random(seed)
    corpus, queries, qrels = {}, {}, {}
    gen = [f"g{i}" for i in range(400)]
    for t in range(n_topics):
        tw = [f"t{t}w{i}" for i in range(25)]
        for j in range(docs_per_topic):
            keys = [f"t{t}d{j}k{m}" for m in range(3)]
            words = keys * 2 + rng.choices(tw, k=20) + rng.choices(gen, k=20)
            words = [w.replace("w", "s") if w.startswith("t") and "w" in w and rng.random() < .15 else w for w in words]
            rng.shuffle(words)
            corpus[f"t{t}d{j}"] = " ".join(words)
    ids = list(corpus)
    for i in range(n_queries):
        d = rng.choice(ids)
        t, j = d[1:].split("d")
        tw = [f"t{t}{'s' if rng.random() < .5 else 'w'}{rng.randrange(25)}" for _ in range(4)]
        words = rng.sample([f"{d}k{m}" for m in range(3)], k=rng.choice([0, 1, 1, 2])) + tw + rng.choices(gen, k=3)
        queries[f"q{i}"] = " ".join(words)
        qrels[f"q{i}"] = {d}
    qs = list(queries)
    return corpus, queries, {q: qrels[q] for q in qs[:350]}, {q: qrels[q] for q in qs[350:]}


# Бэкенды
class STBackend:
    """Реальные модели: sentence-transformers (bi-encoder и cross-encoder)."""
    tag = "st"
    simulated_ms = None

    def __init__(self, device=None):
        self.device, self._enc, self._rr = device, {}, {}

    def encode(self, key, texts, is_query):
        from sentence_transformers import SentenceTransformer
        m = EMBED_MODELS[key]
        if key not in self._enc:
            self._free(self._enc)
            self._enc[key] = SentenceTransformer(m["name"], device=self.device)
        pre = m["q_prefix"] if is_query else m["d_prefix"]
        return self._enc[key].encode([pre + t for t in texts], batch_size=64, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=len(texts) > 1000).astype(np.float32)

    def rerank(self, model, pairs):
        from sentence_transformers import CrossEncoder
        if model not in self._rr:
            self._free(self._rr)
            self._rr[model] = CrossEncoder(RERANKERS[model], max_length=512, device=self.device)
        bs = 8 if model == "bge-m3-rr" else 32
        return np.asarray(self._rr[model].predict(pairs, batch_size=bs, show_progress_bar=False), dtype=np.float32)

    @staticmethod
    def _free(models):
        if models:
            models.clear()
            try:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass


class ToyBackend:
    """Заглушки: LSA вместо эмбеддингов, зашумленный TF-IDF вместо cross-encoder. Латентности заданы вручную."""
    tag = "toy"
    DIMS = {"rubert-tiny2": 48, "me5-small": 128, "mminilm": 96}
    NOISE = {"mminilm-ce": .1, "rubert-ce": .06, "bge-m3-rr": .04}
    simulated_ms = {"bm25": .5, "enc:rubert-tiny2": 2, "enc:me5-small": 6, "enc:mminilm": 6,
                    "rr:mminilm-ce": 4, "rr:rubert-ce": 8, "rr:bge-m3-rr": 25}

    def __init__(self, corpus_texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.tfidf = TfidfVectorizer().fit(corpus_texts)
        X = self.tfidf.transform(corpus_texts)
        self.svd = {k: TruncatedSVD(d, random_state=0).fit(X) for k, d in self.DIMS.items()}

    def encode(self, key, texts, is_query):
        E = self.svd[key].transform(self.tfidf.transform(texts)).astype(np.float32)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def rerank(self, model, pairs):
        q = self.tfidf.transform([p[0] for p in pairs]); d = self.tfidf.transform([p[1] for p in pairs])
        s = np.asarray(q.multiply(d).sum(1)).ravel()
        noise = [np.random.default_rng(zlib.crc32((model + a + b).encode())).normal() for a, b in pairs]
        return (s + self.NOISE[model] * np.array(noise)).astype(np.float32)


# Оценщик
class RAGEvaluator:
    """
    Считает doc-level F1 финального набора документов (как в статье, §3.5) и оценку латентности (мс/запрос).

    Все дорогое кэшируется: эмбеддинги корпуса (на диск), матрицы BM25/dense-скоров,
    скоры cross-encoder для пар (запрос, документ) (на диск). Поэтому один trial занимает
    доли секунды, а «реальную» стоимость мы считаем по модели латентности (cost_ms).
    """

    def __init__(self, corpus, queries, qrels, qids, backend, cache_dir="cache"):
        self.backend, self.cache = backend, Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.doc_ids = list(corpus)
        self.doc_texts = [corpus[d] for d in self.doc_ids]
        idx = {d: i for i, d in enumerate(self.doc_ids)}
        self.qids = [q for q in qids if any(d in idx for d in qrels[q])]
        self.q_texts = [queries[q] for q in self.qids]
        self.rel = [{idx[d] for d in qrels[q] if d in idx} for q in self.qids]

        cv = CountVectorizer(analyzer=ru_analyzer)
        tf = cv.fit_transform(self.doc_texts).tocsr().astype(np.float32)
        self._tf = tf
        self._rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        dl = np.asarray(tf.sum(1)).ravel()
        self._dl_norm = (dl / dl.mean())[self._rows]
        df = np.asarray((tf > 0).sum(0)).ravel()
        N = tf.shape[0]
        self._idf = np.log(1 + (N - df + .5) / (df + .5)).astype(np.float32)
        self._Qb = (cv.transform(self.q_texts) > 0).astype(np.float32)

        self.lat = dict(backend.simulated_ms or {})
        self._bm25_lru, self._dense, self._memo = OrderedDict(), {}, {}
        self._rr = {m: self._load_rr(m) for m in RERANKERS}

    # retrievers
    def _bm25(self, k1, b):
        key = (k1, b)
        if key in self._bm25_lru:
            self._bm25_lru.move_to_end(key)
            return self._bm25_lru[key]
        t0 = time.perf_counter()
        d = self._tf.data
        w = self._idf[self._tf.indices] * d * (k1 + 1) / (d + k1 * (1 - b + b * self._dl_norm))
        W = csr_matrix((w, self._tf.indices, self._tf.indptr), shape=self._tf.shape)
        S = (self._Qb @ W.T).toarray().astype(np.float32)
        self.lat.setdefault("bm25", 1000 * (time.perf_counter() - t0) / len(self.qids))
        self._bm25_lru[key] = S
        if len(self._bm25_lru) > 64:
            self._bm25_lru.popitem(last=False)
        return S

    def _dense_scores(self, key):
        if key not in self._dense:
            p = self.cache / f"emb_{self.backend.tag}_{key}.npy"
            if p.exists() and np.load(p, mmap_mode="r").shape[0] == len(self.doc_ids):
                D = np.load(p)
            else:
                print(f"  кодирую корпус моделью {key}...")
                D = self.backend.encode(key, self.doc_texts, is_query=False)
                np.save(p, D)
            t0 = time.perf_counter()
            Q = self.backend.encode(key, self.q_texts, is_query=True)
            self.lat.setdefault(f"enc:{key}", 1000 * (time.perf_counter() - t0) / len(self.qids))
            self._dense[key] = Q @ D.T
        return self._dense[key]

    @staticmethod
    def _minmax(S):
        mn = S.min(1, keepdims=True)
        return (S - mn) / (S.max(1, keepdims=True) - mn + 1e-9)

    def _retrieval_scores(self, c):
        if c["retriever"] == "bm25":
            return self._bm25(c["k1"], c["b"])
        if c["retriever"] == "dense":
            return self._dense_scores(c["embed"])
        a = c["alpha"]
        return a * self._minmax(self._dense_scores(c["embed"])) + (1 - a) * self._minmax(self._bm25(c["k1"], c["b"]))

    # reranker
    def _rr_path(self, m):
        return self.cache / f"rr_{self.backend.tag}_{m}.pkl"

    def _load_rr(self, m):
        p = self._rr_path(m)
        if p.exists():
            with open(p, "rb") as f:
                data = pickle.load(f)
            if "lat" in data:
                self.lat.setdefault(f"rr:{m}", data["lat"])
            return data["scores"]
        return {}

    def save(self):
        for m, sc in self._rr.items():
            if sc:
                with open(self._rr_path(m), "wb") as f:
                    pickle.dump({"scores": sc, "lat": self.lat.get(f"rr:{m}")}, f)

    def _rerank(self, m, cand):
        cache = self._rr[m]
        todo = sorted({(i, j) for i, row in enumerate(cand) for j in row if (self.qids[i], j) not in cache})
        if todo:
            t0 = time.perf_counter()
            s = self.backend.rerank(m, [(self.q_texts[i], self.doc_texts[j]) for i, j in todo])
            if f"rr:{m}" not in self.lat:
                self.lat[f"rr:{m}"] = 1000 * (time.perf_counter() - t0) / len(todo)
            for (i, j), v in zip(todo, s):
                cache[(self.qids[i], j)] = float(v)
        return [np.array([cache[(self.qids[i], j)] for j in row], dtype=np.float32) for i, row in enumerate(cand)]

    # Основной вызов
    def retrieve(self, cfg):
        """Итоговые документы (индексы в self.doc_ids) для каждого запроса из self.qids."""
        c = normalize(cfg)
        S = self._retrieval_scores(c)
        k = min(c["top_k"], S.shape[1])
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(len(S))[:, None]
        order = np.argsort(-S[rows, part], axis=1)
        cand = part[rows, order]
        scores = [S[i, cand[i]] for i in range(len(S))]
        cand = list(cand)

        if c["reranker"] != "none":
            rs = self._rerank(c["reranker"], cand)
            new_c, new_s = [], []
            for row, s in zip(cand, rs):
                o = np.argsort(-s)[: c["rerank_top_k"]]
                new_c.append(row[o]); new_s.append(s[o])
            cand, scores = new_c, new_s

        if c["filter"] == "relative":
            out = []
            for row, s in zip(cand, scores):
                n = (s - s.min()) / (s.max() - s.min() + 1e-9) if len(s) > 1 else np.ones(1)
                out.append(row[n >= c["filter_thr"]])
            cand = out
        return cand

    def evaluate(self, cfg):
        c = normalize(cfg)
        key = cfg_key(c)
        if key in self._memo:
            return self._memo[key]
        cand = self.retrieve(c)

        P = R = F = 0.0
        for row, rel in zip(cand, self.rel):
            hit = len(set(row.tolist()) & rel)
            p, r = hit / max(len(row), 1), hit / len(rel)
            P, R, F = P + p, R + r, F + (2 * p * r / (p + r) if hit else 0)
        n = len(self.rel)
        res = dict(f1=F / n, precision=P / n, recall=R / n, cost_ms=self.cost_ms(c))
        self._memo[key] = res
        return res

    def cost_ms(self, c):
        """Оценка латентности одного запроса через выбранную конфигурацию."""
        ms = 0.0
        if c["retriever"] in ("bm25", "hybrid"):
            ms += self.lat.get("bm25", 0)
        if c["retriever"] in ("dense", "hybrid"):
            ms += self.lat.get(f"enc:{c['embed']}", 0)
        if c["reranker"] != "none":
            ms += self.lat.get(f"rr:{c['reranker']}", 0) * c["top_k"]
        return ms
