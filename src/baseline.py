"""TF-IDF + Logistic Regression baseline for AG News classification."""
import json
import time

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_pipeline import LABEL_NAMES


def main():
    raw = load_dataset("fancyzhx/ag_news")
    X_train, y_train = raw["train"]["text"], raw["train"]["label"]
    X_test, y_test = raw["test"]["text"], raw["test"]["label"]

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    start = time.time()
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_vec, y_train)
    train_time = time.time() - start

    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    report = classification_report(y_test, preds, target_names=LABEL_NAMES, output_dict=True)

    results = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "train_time_sec": train_time,
        "per_class_report": report,
    }
    print(json.dumps({k: v for k, v in results.items() if k != "per_class_report"}, indent=2))

    with open("../reports/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
