"""
train.py — Fine-tune BERT for fake news classification.

Dataset expected: CSV with columns ['text', 'label']
  where label = 0 (REAL) or 1 (FAKE).

Popular datasets to use:
  - LIAR dataset:     https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
  - FakeNewsNet:      https://github.com/KaiDMML/FakeNewsNet
  - WELFake:          https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
  - ISOT Fake News:   https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php

Usage:
    python train.py --data_path data/news.csv --output_dir saved_model/ --epochs 3
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Dataset Class ────────────────────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    """PyTorch Dataset for tokenized news articles."""

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: BertTokenizer,
        max_len: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, accuracy, auc, all_preds, all_labels


# ─── Plot Helpers ─────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(history["val_acc"], label="Val Acc", marker="s")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    logger.info(f"Training curves saved to {path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    logger.info(f"Confusion matrix saved to {path}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Load Data ──
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    assert "text" in df.columns and "label" in df.columns, \
        "CSV must have 'text' and 'label' columns"

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    logger.info(f"Total samples: {len(df)} | Real: {(df.label==0).sum()} | Fake: {(df.label==1).sum()}")

    # ── Train/Val Split ──
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.15,
        random_state=42,
        stratify=df["label"].tolist(),
    )

    # ── Tokenizer & Model ──
    logger.info("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    model.to(device)

    # ── Datasets & Loaders ──
    train_ds = FakeNewsDataset(X_train, y_train, tokenizer, args.max_len)
    val_ds = FakeNewsDataset(X_val, y_val, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Optimizer & Scheduler ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Training ──
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*50}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_auc, val_preds, val_labels = eval_epoch(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val   Loss: {val_loss:.4f}  | Val Acc:   {val_acc:.4f} | AUC: {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.info(f"✅ Best model saved (val_acc={val_acc:.4f})")

    # ── Final Evaluation ──
    logger.info("\n📊 Final Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=["REAL", "FAKE"]))

    # ── Save Artifacts ──
    plot_training_curves(history, args.output_dir)
    plot_confusion_matrix(val_labels, val_preds, args.output_dir)

    metrics = {
        "best_val_accuracy": best_val_acc,
        "final_val_auc": val_auc,
        "epochs": args.epochs,
        "model_name": args.model_name,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\n✅ Training complete. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for fake news detection")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--output_dir", type=str, default="saved_model/", help="Where to save the model")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    main(args)
