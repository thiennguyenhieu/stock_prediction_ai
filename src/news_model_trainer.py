import pandas as pd
from datasets import Dataset
from underthesea import word_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

# === CONFIG ===
MODEL_NAME = "./models/models--vinai--phobert-base"  # Local path to PhoBERT
OUTPUT_DIR = "./models/phobert-finance"
NUM_LABELS = 3
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CSV_PATH = "data/headlines.csv"  # Fixed path to training data

# === FUNCTIONS ===
def label_by_keywords(text):
    text = text.lower()
    positive_kw = ["lãi", "tăng", "khởi sắc", "bứt phá", "mua vào", "hồi phục", "dương"]
    negative_kw = ["giảm", "lỗ", "bán tháo", "đỏ lửa", "lao dốc", "rớt giá", "âm", "sụt giảm"]

    if any(kw in text for kw in positive_kw):
        return "positive"
    elif any(kw in text for kw in negative_kw):
        return "negative"
    return "neutral"

def vi_tokenize(text):
    return word_tokenize(text, format="text")

def preprocess_function(examples):
    examples['text'] = vi_tokenize(examples['text'])
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# === LOAD DATA ===
def load_dataset_from_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['text'])

    # Auto-label if 'label' column is missing
    if 'label' not in df.columns:
        print("\n🧠 Auto-labeling using keywords...")
        df['label'] = df['text'].apply(label_by_keywords)

    df['label'] = df['label'].map(LABEL2ID)
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=0.2)

# === MAIN TRAINING PIPELINE ===
if __name__ == "__main__":
    print("\n📥 Loading dataset from:", CSV_PATH)
    raw_dataset = load_dataset_from_csv(CSV_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    print("\n🔄 Tokenizing data...")
    tokenized_train = raw_dataset["train"].map(preprocess_function)
    tokenized_eval = raw_dataset["test"].map(preprocess_function)

    print("\n📦 Loading PhoBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    print("\n⚙️ Setting training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
        load_best_model_at_end=True
    )

    print("\n🏋️ Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("\n✅ Training complete. Model saved to:", OUTPUT_DIR)
