"""Gradio demo for the fine-tuned BERT news classifier."""
import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "../models/bert-ag-news"
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

EXAMPLES = [
    "The central bank raised interest rates again today, citing persistent inflation pressures across the eurozone.",
    "The home team clinched the championship in overtime after a last-second three-pointer.",
    "Researchers unveiled a new chip architecture that promises double the performance per watt.",
    "Diplomats from both nations met in Geneva to discuss a ceasefire agreement.",
]


@torch.no_grad()
def classify(text: str):
    if not text or not text.strip():
        return {}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    return {label: prob for label, prob in zip(LABEL_NAMES, probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, label="News headline / article snippet", placeholder="Paste a news headline or short article..."),
    outputs=gr.Label(num_top_classes=4, label="Predicted category"),
    examples=EXAMPLES,
    title="AG News Classifier (fine-tuned BERT)",
    description=(
        "bert-base-uncased fine-tuned on the AG News dataset. "
        "Test accuracy: 94.7% | macro-F1: 94.7% (see reports/bert_results.json)."
    ),
)

if __name__ == "__main__":
    demo.launch()
