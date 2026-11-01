"""Fine-tune BERT-base on AG News with HuggingFace Trainer.

Note: fp16 mixed precision (as used on CUDA/Colab T4) is not used here since
this runs on Apple Silicon MPS, which does not support torch fp16 autocast
the same way. Training otherwise mirrors the original recipe: cosine warmup,
weight decay, 128-token truncation.
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from data_pipeline import load_tokenized_dataset, LABEL_NAMES, MODEL_NAME


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenized, tokenizer = load_tokenized_dataset()

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

    args = TrainingArguments(
        output_dir="../models/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        warmup_steps=1500,  # ~10% of total steps (120000/16 * 2 epochs = 15000)
        lr_scheduler_type="cosine",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    eval_results = trainer.evaluate()

    preds_output = trainer.predict(tokenized["test"])
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    report = classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()

    final_results = {
        "model": "bert-base-uncased (fine-tuned)",
        "device": device,
        "train_time_sec": train_time,
        "accuracy": eval_results["eval_accuracy"],
        "macro_f1": eval_results["eval_macro_f1"],
        "per_class_report": report,
        "confusion_matrix": cm,
        "label_names": LABEL_NAMES,
    }

    with open("../reports/bert_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    model.save_pretrained("../models/bert-ag-news")
    tokenizer.save_pretrained("../models/bert-ag-news")

    print(json.dumps({k: v for k, v in final_results.items() if k not in ("per_class_report", "confusion_matrix")}, indent=2))
    print("Saved model to ../models/bert-ag-news")


if __name__ == "__main__":
    main()
