import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune a transformer on the local sentiment dataset and generate result.csv."
    )
    parser.add_argument(
        "--data_dir",
        default="project-sentiment-analysis/local-datasets",
        help="Directory containing train.tsv, test.tsv, sampleSubmission.csv.",
    )
    parser.add_argument(
        "--model_name",
        default="distilroberta-base",
        help="Hugging Face model checkpoint to start from.",
    )
    parser.add_argument(
        "--output_dir",
        default="project-sentiment-analysis/model-output",
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument(
        "--result_path",
        default="project-sentiment-analysis/result.csv",
        help="Where to write the submission CSV.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow loading checkpoints whose classifier head shape differs from num_labels (default: True).",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
        ],
        help="Learning rate scheduler type.",
    )
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps.")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio over total training steps (use instead of warmup_steps).",
    )
    parser.add_argument(
        "--eval_fraction",
        type=float,
        default=0.1,
        help="Portion of training data for validation.",
    )
    return parser.parse_args()


def load_local_dataset(data_dir: str):
    data_files = {
        "train": os.path.join(data_dir, "train.tsv"),
        "test": os.path.join(data_dir, "test.tsv"),
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    dataset = dataset.rename_column("Phrase", "text")
    dataset = dataset.cast_column("PhraseId", dataset["train"].features["PhraseId"])
    dataset = dataset.cast_column("SentenceId", dataset["train"].features["SentenceId"])
    train_split = dataset["train"].rename_column("Sentiment", "label")
    train_split = train_split.map(lambda x: {"label": int(x["label"])})
    train_split = train_split.class_encode_column("label")
    dataset["train"] = train_split
    return dataset


def tokenize_datasets(dataset, tokenizer):
    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        texts = ["" if t is None else str(t) for t in batch["text"]]
        return tokenizer(texts, truncation=True)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def build_label_maps(num_labels: int) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2label = {i: str(i) for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.warmup_steps > 0 and args.warmup_ratio > 0:
        raise ValueError("Specify only one of warmup_steps or warmup_ratio.")

    raw_dataset = load_local_dataset(args.data_dir)
    label_list = raw_dataset["train"].features["label"].names
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    id2label, label2id = build_label_maps(num_labels)

    split_dataset = raw_dataset["train"].train_test_split(
        test_size=args.eval_fraction,
        seed=args.seed,
        stratify_by_column="label",
    )
    tokenized_train = tokenize_datasets(split_dataset["train"], tokenizer)
    tokenized_eval = tokenize_datasets(split_dataset["test"], tokenizer)
    tokenized_test = tokenize_datasets(raw_dataset["test"], tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_strategy="steps",
        logging_steps=50,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    print(
        f"Validation metrics: accuracy={eval_metrics.get('eval_accuracy'):.4f}, "
        f"macro_f1={eval_metrics.get('eval_macro_f1'):.4f}"
    )

    # Persist the finetuned model and tokenizer for reuse.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    predictions = trainer.predict(tokenized_test).predictions
    pred_labels = np.argmax(predictions, axis=-1)

    submission = pd.DataFrame(
        {
            "PhraseId": raw_dataset["test"]["PhraseId"],
            "Sentiment": pred_labels,
        }
    )
    submission.to_csv(args.result_path, index=False)
    print(f"Saved submission to {args.result_path}")


if __name__ == "__main__":
    main()
