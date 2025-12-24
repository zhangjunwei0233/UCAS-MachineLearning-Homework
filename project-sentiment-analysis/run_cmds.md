# First run

```bash
uv run python project-sentiment-analysis/run_sentiment.py \
  --model_name distilbert-base-cased \
  --num_train_epochs 2 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32
```

lr = 2e-5
linear scheduler, no warmup

Score: 0.69195


# Second run

```bash
uv run python project-sentiment-analysis/run_sentiment.py \
  --model_name distilroberta-base \
  --num_train_epochs 4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1
```