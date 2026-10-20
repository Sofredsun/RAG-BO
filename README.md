# Байесовская оптимизация против grid/random search для настройки RAG

Упрощенная проверка идей статьи Afzal et al., *Beyond Grid Search: Leveraging Bayesian Optimization for
Accelerating RAG Pipeline Optimization* (EACL 2026, Industry Track) на русском датасете rus-SciFact
(перевод SciFact из бенчмарка RusBEIR, Kovalev et al., 2025; HuggingFace: `kaengreg/rus-scifact`).

## Что воспроизводится
| Статья | Здесь                                                                   |
|---|-------------------------------------------------------------------------|
| Пайплайн AutoRAG: 6 стадий | 3 стадии: retriever -> reranker -> filter (без генерации)               |
| Grid search (локальный, покомпонентный) как бейзлайн | то же                                                                   |
| Random search, Optuna TPE, SMAC3 (Random Forest) | random search и Optuna TPE (SMAC3, GP/HEBO и multi-fidelity не включены) |
| Локальная vs глобальная оптимизация | то же, с одинаковым общим бюджетом (по умолчанию 60 = 3×20 против 60) |
| Doc-level F1 для retrieval-компонент | то же, это и есть целевая метрика                                       |
| Парето-фронт «качество — латентность» | то же, латентность из модели стоимости (см. ниже)                       |

Пространство поиска (`space.py`): тип retriever (BM25 с русским стеммингом / dense / hybrid),
3 русско- или многоязычные модели эмбеддингов (rubert-tiny2, multilingual-e5-small, multilingual MiniLM), `k1`, `b`,
вес гибрида `alpha`, `top_k` (5–30), 3 cross-encoder реранкера (mMiniLM-mMARCO, rubert-msmarco,
bge-reranker-v2-m3) или без него, `rerank_top_k` (≤ `top_k`,
как ограничение в Appendix D статьи), фильтр по относительному порогу скора. Смешанное пространство
с условными параметрами — ровно тот случай, ради которого в статье нужен BO.

## Запуск
```bash
pip install -r requirements.txt
python run.py --data toy --seeds 0 --budget 30      # Проверка без скачивания моделей
python run.py --data rus-scifact --seeds 0 1 2 --budget 60 --device cuda
python plot.py --data rus-scifact
```
Результаты: `results/trials_rus-scifact.csv` (все trials), `results/final_rus-scifact.csv` (лучший конфиг каждого
запуска + его F1 на отложенном тесте), `summary_*.csv`, `convergence_*.png`, `pareto_*.png`.

Оптимизация идет на 150 запросах из train-разметки rus-SciFact, итоговая проверка - на 300 test-запросах.

## Примеры ответов LLM
```bash
python answer.py --data rus-scifact --device cuda --n 5
# Крупнее ~3.5 ГБ видеопамяти:
python answer.py --data rus-scifact --device cuda --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16
```
Берет лучшую конфигурацию из `results/final_rus-scifact.csv` (выбор по валидационному F1) и бейзлайн
(BM25, top_k=3, без реранкера), находит документы для нескольких test-запросов и просит маленькую LLM
оценить утверждение: SUPPORTS / REFUTES / NOT ENOUGH INFO со ссылкой на документ. Результат —
`results/answers_rus-scifact.md`: найденные документы с пометкой ✓/✗ по разметке и ответ модели.
Это иллюстрация, а не метрика: генерация не оптимизировалась и не оценивалась.

## Как измеряется время
Благодаря кэшу один trial идет доли секунды, поэтому реальное время прогона не отражает стоимость
оптимизации. Вместо него используется модель стоимости: латентность каждой компоненты
(BM25 на запрос, кодирование запроса, cross-encoder на пару) измеряется один раз, а стоимость trial =
латентность конфигурации × число запросов. Сумма по trials — «сколько шла бы оптимизация без кэша»
(`opt_time_min`). Это аналог колонки Time в таблицах статьи, но оценочный.

## Ограничения
- Нет SMAC3 (его зависимость pyrfr плохо ставится на Windows): BO представлен только TPE.
- Нет генератора и LLM-метрик: проверяется только retrieval-часть RAG.
- Маленькое пространство, поэтому grid search может оказаться конкурентным.
- В локальной оптимизации первые 20 trials оптимизируют только retriever (reranker выключен),
  поэтому их F1 ниже по построению — это не баг, а следствие покомпонентной схемы.
- Результаты на `--data toy` синтетические и ничего не говорят о методах, это только тест кода.
