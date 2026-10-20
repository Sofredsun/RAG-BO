import argparse, json, random
from pathlib import Path

import pandas as pd

from pipeline import RAGEvaluator, STBackend, ToyBackend, load_rus_scifact, make_toy
from space import normalize

BASELINE = dict(retriever="bm25", k1=1.2, b=0.75, top_k=3, reranker="none", filter="none")

PROMPTS = {
    "verify": """Ты проверяешь научные утверждения, используя только документы ниже.

Утверждение: {claim}

Документы:
{docs}

Ответь одной строкой на русском: начни с ПОДТВЕРЖДАЕТ, ОПРОВЕРГАЕТ или НЕДОСТАТОЧНО ДАННЫХ, \
затем одно предложение обоснования со ссылкой на номер документа, например [1].""",
    "summarize": """Ниже научное утверждение и найденные для него документы.

Утверждение: {claim}

Документы:
{docs}

Отвечай только на русском. В 2–3 предложениях перескажи, что документы сообщают по поводу утверждения, \
со ссылками на номера документов, например [1]. Внимательно учитывай отрицания в утверждении. \
Затем дай итоговую оценку одним словом: ПОДТВЕРЖДАЕТ, ОПРОВЕРГАЕТ или НЕОДНОЗНАЧНО. \
Если ни один документ не относится к утверждению, так и напиши.""",
}
PROMPT = PROMPTS["verify"]


def build_prompt(claim, texts, doc_chars=2000, task="summarize"):
    ctx = [f"[{n}] {t[:doc_chars]}" for n, t in enumerate(texts, 1)]
    return PROMPTS[task].format(claim=claim, docs="\n\n".join(ctx)
    if ctx else "(документов нет)")


class Generator:
    def __init__(self, model, device, dtype, max_new_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        try:
            m = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype))
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=getattr(torch, dtype))
        self.model = m.to(device).eval()
        self.device, self.max_new_tokens = device, max_new_tokens

    def __call__(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def pick_best(final_csv, method, scope):
    f = pd.read_csv(final_csv)
    if method:
        f = f[f.method == method]
    if scope:
        f = f[f.scope == scope]
    if f.empty:
        raise SystemExit("Нет подходящих запусков в " + str(final_csv) + ": сначала запустите run.py")
    row = f.sort_values("val_f1", ascending=False).iloc[0]
    return f"{row.scope} / {row.method} (seed {row.seed})", json.loads(row.cfg)


def fmt_cfg(c):
    c = normalize(c)
    parts = [c["retriever"] + (f"({c['embed']})" if c["embed"] else ""), f"top_k={c['top_k']}"]
    if c["reranker"] != "none":
        parts.append(f"rerank={c['reranker']}@{c['rerank_top_k']}")
    if c["filter"] != "none":
        parts.append(f"filter≥{c['filter_thr']}")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rus-scifact", choices=["rus-scifact", "toy"])
    ap.add_argument("--n", type=int, default=5, help="сколько test-запросов показать")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default=None, help="взять лучший конфиг только этого метода (grid/random/tpe)")
    ap.add_argument("--scope", default=None, choices=[None, "local", "global"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--doc-chars", type=int, default=2000, help="обрезка каждого документа в промпте")
    ap.add_argument("--task", default="summarize", choices=list(PROMPTS),
                    help="summarize — пересказ доказательств и оценка; verify — строгий вердикт")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true", help="не загружать LLM, только поиск и промпты")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    corpus, queries, _, qr_test = load_rus_scifact() if a.data == "rus-scifact" else make_toy()
    backend = STBackend(a.device) if a.data == "rus-scifact" else ToyBackend(list(corpus.values()))
    ev = RAGEvaluator(corpus, queries, qr_test, sorted(qr_test), backend, cache_dir=f"cache/{a.data}")

    out = Path(a.out)
    best_name, best_cfg = pick_best(out / f"final_{a.data}.csv", a.method, a.scope)
    configs = [(f"Лучшая найденная: {best_name}", best_cfg), ("Бейзлайн без настройки", BASELINE)]
    retrieved = {name: ev.retrieve(cfg) for name, cfg in configs}
    scores = {name: ev.evaluate(cfg)["f1"] for name, cfg in configs}

    idx = sorted(random.Random(a.seed).sample(range(len(ev.qids)), min(a.n, len(ev.qids))))
    gen = None if a.dry_run else Generator(a.model, a.device, a.dtype, a.max_new_tokens)

    lines = [f"# Примеры ответов ({a.data})", "",
             f"Генератор: {'— (dry-run)' if gen is None else a.model + ', ' + a.dtype}, режим {a.task}. "
             "Ответы иллюстративные: генерация не оптимизировалась и не оценивалась.", ""]
    for name, cfg in configs:
        lines.append(f"- **{name}**: `{fmt_cfg(cfg)}`, doc-F1 на test = {scores[name]:.3f}")
    for qi in idx:
        claim = ev.q_texts[qi]
        lines += ["", "---", "", f"## Утверждение: {claim}", ""]
        for name, _ in configs:
            docs = retrieved[name][qi].tolist()
            hits = sum(d in ev.rel[qi] for d in docs)
            lines.append(f"**{name}** — найдено {len(docs)}, из них релевантных по разметке: {hits}/{len(ev.rel[qi])}")
            lines.append("")
            for n, d in enumerate(docs, 1):
                mark = "✓" if d in ev.rel[qi] else "✗"
                lines.append(f"{n}. {mark} {ev.doc_texts[d][:150]}…")
            prompt = build_prompt(claim, [ev.doc_texts[d] for d in docs], a.doc_chars, a.task)
            answer = "(dry-run: ответ не генерировался)" if gen is None else gen(prompt)
            quoted = "\n> ".join(answer.splitlines())
            lines += ["", f"> **Ответ LLM:** {quoted}", ""]
        print(f"готово: {claim[:70]}")
    ev.save()
    path = out / f"answers_{a.data}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("Сохранено:", path)


if __name__ == "__main__":
    main()
