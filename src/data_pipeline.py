"""Load and tokenize the AG News dataset for BERT fine-tuning."""
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def load_tokenized_dataset(tokenizer_name: str = MODEL_NAME, max_length: int = MAX_LENGTH):
    raw = load_dataset("fancyzhx/ag_news")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized, tokenizer


if __name__ == "__main__":
    ds, tok = load_tokenized_dataset()
    print(ds)
    print("Train size:", len(ds["train"]), "Test size:", len(ds["test"]))
