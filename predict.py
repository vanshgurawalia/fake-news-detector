"""
predict.py — CLI inference script for the trained FakeShield model.

Usage:
    # Single text
    python predict.py --text "NASA discovers water on Mars"

    # Batch from CSV
    python predict.py --csv_path data/test.csv --output_path predictions.csv

    # Load fine-tuned model
    python predict.py --model_dir saved_model/ --text "Breaking: Secret exposed!"
"""

import argparse
import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from utils import preprocess_text, get_confidence_label, get_credibility_score


def load_model(model_dir: str = "bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    model.eval()
    return tokenizer, model


def predict_single(text: str, tokenizer, model, max_len: int = 256, device="cpu"):
    cleaned = preprocess_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    real_prob, fake_prob = float(probs[0]), float(probs[1])
    label = "FAKE" if fake_prob > real_prob else "REAL"
    conf_label, _ = get_confidence_label(fake_prob if label == "FAKE" else real_prob)
    cred = get_credibility_score(text, real_prob)

    return {
        "label": label,
        "real_probability": round(real_prob, 4),
        "fake_probability": round(fake_prob, 4),
        "confidence": conf_label,
        "credibility_score": cred,
    }


def predict_batch(texts: list, tokenizer, model, max_len: int = 256, device="cpu"):
    results = []
    for text in texts:
        result = predict_single(text, tokenizer, model, max_len, device)
        result["text"] = text[:120] + "..." if len(text) > 120 else text
        results.append(result)
    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {args.model_dir}")
    tokenizer, model = load_model(args.model_dir)
    model.to(device)

    if args.text:
        result = predict_single(args.text, tokenizer, model, args.max_len, device)
        print("\n" + "=" * 50)
        print("🛡️  FakeShield Prediction")
        print("=" * 50)
        print(f"  Text       : {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
        print(f"  Verdict    : {result['label']}")
        print(f"  Real Prob  : {result['real_probability']:.4f}")
        print(f"  Fake Prob  : {result['fake_probability']:.4f}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  Credibility: {result['credibility_score']}/10")
        print("=" * 50)

    elif args.csv_path:
        df = pd.read_csv(args.csv_path)
        assert "text" in df.columns, "CSV must have a 'text' column"
        print(f"Running batch prediction on {len(df)} samples...")
        results = predict_batch(df["text"].tolist(), tokenizer, model, args.max_len, device)
        out_df = pd.DataFrame(results)
        out_path = args.output_path or "predictions.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\n✅ Predictions saved to: {out_path}")
        print(out_df[["label", "real_probability", "fake_probability", "credibility_score"]].describe())

    else:
        print("Provide --text or --csv_path. Run with --help for usage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Single text to classify")
    parser.add_argument("--csv_path", type=str, default=None, help="CSV file with 'text' column")
    parser.add_argument("--output_path", type=str, default="predictions.csv")
    parser.add_argument("--model_dir", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()
    main(args)
