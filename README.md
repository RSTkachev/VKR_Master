# Выпускная квалификационная работа

Репозиторий содержит код к ВКР Ткачева Радомира Святославовича. Реализован двухстадийный поиск текстов по изображению для каталога товаров [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html):

1. **Стадия 1 — двухбашенный энкодер.** CLIP / SigLIP / SigLIP2 (опционально с LoRA-адаптацией) кодирует изображения и тексты в общее L2-нормализованное пространство. Поиск кандидатов выполняется индексом HNSW из библиотеки FAISS.
2. **Стадия 2 — переранжирование с помощью VLM.** Qwen3.5-4B (опционально с LoRA-адаптацией) переупорядочивает первые K кандидатов одним из двух способов: pointwise (по логитам ответа Yes / No) или listwise (генерация полного ранжирования за один проход).

Качество измеряется на фиксированных разбиениях датасета ABO.

## Структура

```
.
├── scripts/                            # скрипты запуска экспериментов
├── src/vkr/                            # исходный код пакета vkr
│   ├── ann.py                          # поиск ближайших соседей через FAISS HNSW
│   ├── bench_model.py                  # оценка качества первой и (опционально) второй стадии
│   ├── encode.py                       # пакетное кодирование текстов и изображений
│   ├── finetune_dualencoder.py         # обучение SigLIP c InfoNCE / sigmoid + LoRA
│   ├── finetune_listwise.py            # обучение Qwen в режиме listwise-переранжирования
│   ├── preprocessing.py                # парсинг и обработка метаданных ABO
│   ├── reranking.py                    # общие шаблоны запросов и вспомогательные функции для переранжирования
│   ├── scoring.py                      # метрики recall@K и MRR
│   ├── splits.py                       # работа с разбиениями данных
│   ├── utils.py                        # служебные функции
│   ├── analyze_siglip_similarity.py    # анализ распределений векторных представлений
│   └── analyze_rerank_errors.py        # сравнение результатов первой и второй стадии
├── pyproject.toml
└── README.md
```

## Установка

```bash
pip install -e .
# Опционально — flash-attention для bfloat16 на CUDA:
pip install flash-attn --no-build-isolation
```

Требуется Python ≥ 3.11. Полный список зависимостей — в [`pyproject.toml`](pyproject.toml).

## Данные

Скачайте `abo-listings.tar` и `abo-images-small.tar` со страницы [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) и распакуйте в:

```
data/raw_data/abo/
├── images/
│   ├── metadata/images.csv
│   └── small/...
└── metadata/*.json
```

## Воспроизведение экспериментов

### 1. Создать фиксированное разбиение

```bash
python -m vkr.splits create \
    --name abo_v4 \
    --train-siglip 14500 \
    --train-reranker 14500 \
    --val 10000 \
    --test 10000 \
    --preferred-lang en \
    --strict-preferred-lang \
    --dedupe-by-main-image \
    --unique-other-images-only \
    --subsample-category "CELLULAR_PHONE_CASE:5000" \
    --seed 42
```

Манифест разбиения (`data/splits/abo_v4/`) хранит список `item_id` и параметры предобработки — впоследствии `load_split` детерминированно восстанавливает те же объекты `Example`.

### 2. Обучить первую стадию (SigLIP + LoRA)

```bash
python -m vkr.finetune_dualencoder \
    --split-name abo_v4 \
    --model_id google/siglip2-so400m-patch14-384 \
    --use-lora \
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
    --lora-target-modules q_proj,k_proj,v_proj,out_proj \
    --loss-function sigmoid \
    --use-multi-positive \
    --main-image-only \
    --learning-rate 2e-4 \
    --weight-decay 0.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --scheduler cosine --warmup-ratio 0.05 \
    --train-batch-size 20 --infer-batch-size 256 \
    --epochs 20 --patience 5 --min-delta 0.001 \
    --seed 42 --device cuda \
    --checkpoint-dir ./checkpoints \
    --run-name <run_name>
```

Дополнительные варианты для сравнительных запусков — в [`scripts/finetune_dualencoder.sh`](scripts/finetune_dualencoder.sh).

### 3. Обучить вторую стадию (Qwen, listwise, LoRA)

```bash
python -m vkr.finetune_listwise \
    --split-name abo_v4 \
    --siglip-model-id google/siglip2-so400m-patch14-384 \
    --siglip-lora ./checkpoints/<siglip_lora> \
    --vlm-model-id Qwen/Qwen3.5-4B \
    --top-k 10 \
    --epochs 10 --patience 3 --min-delta 0.005 \
    --learning-rate 6e-5 \
    --weight-decay 0.05 \
    --lora-rank 32 --lora-alpha 64 \
    --lora-dropout 0.1 \
    --train-batch-size 1 \
    --grad-accum-steps 16 \
    --max-eval-examples 1000 \
    --output-dir ./checkpoints/<qwen_lora> \
    --regenerate-pairs \
    --seed 42 --device cuda \
    --run-name <run_name>
```

### 4. Итоговая оценка на тестовом разбиении

```bash
python -m vkr.bench_model \
    --split-name abo_v4 --split-part test \
    --model siglip2 \
    --model_id google/siglip2-so400m-patch14-384 \
    --lora-checkpoint ./checkpoints/<siglip_lora> \
    --top-k 10 \
    --second_model_id Qwen/Qwen3.5-4B \
    --rerank-mode listwise \
    --second-lora-checkpoint ./checkpoints/<qwen_lora> \
    --save-per-query-ranks \
    --run-name <run_name>
```

Выходные файлы:
- `eval_results/<...>.json` — агрегированные метрики (R@1 / R@5 / R@10, MRR, задержки, потребление памяти).
- `eval_results/<...>.npz` — ранги в разрезе запросов (`ranks`, `retriever_ranks`, `item_ids`). Создаётся только при указании `--save-per-query-ranks`.

### 5. Анализ ошибок переранжирования

```bash
python -m vkr.analyze_rerank_errors \
    --retriever-result ./eval_results/<retriever_only>.npz \
    --reranker-result  ./eval_results/<retriever_plus_reranker>.npz \
    --split-name abo_v4 --split-part test \
    --out-dir ./error_analysis
```

Скрипт строит матрицу ошибок, распределение сдвигов рангов и разрез по категориям товаров (`product_type`).

## Замечания

- Тестовое разбиение используется **только** для итоговой оценки; настройка гиперпараметров производится на валидационной части.
- Все адаптеры LoRA сохраняются через `PeftModel.save_pretrained` и могут быть слиты в базовую модель вызовом `merge_and_unload()` перед применением.
- Адрес MLflow читается из переменной окружения `MLFLOW_TRACKING_URI` (можно положить в `.env`).
