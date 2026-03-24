"""
utils.py — helper functions for FakeShield fake news detector.
"""

import re
import string
from typing import Tuple, Dict, List


# ─── Suspicious Pattern Dictionary ────────────────────────────────────────────
SUSPICIOUS_PATTERNS: Dict[str, List[str]] = {
    "Sensationalism": [
        r"\bbreaking\b",
        r"\bexclusive\b",
        r"\bshocking\b",
        r"\bexplosive\b",
        r"\bscandalous\b",
        r"\boutrageous\b",
        r"\bstunning\b",
    ],
    "Clickbait": [
        r"you won't believe",
        r"what (they|he|she) don't want you to know",
        r"the truth about",
        r"they don't want",
        r"mainstream media",
        r"wake up",
        r"\bexposed\b",
    ],
    "Conspiracy Language": [
        r"\bcover.?up\b",
        r"\bdeep state\b",
        r"\bnew world order\b",
        r"\bcabal\b",
        r"\bplandemic\b",
        r"\bscam\b.*\bgovernment\b",
        r"\bthey.re hiding\b",
    ],
    "Excessive Caps / Punctuation": [
        r"[A-Z]{4,}",         # 4+ consecutive uppercase letters
        r"!{2,}",             # Multiple exclamation marks
        r"\?{2,}",            # Multiple question marks
        r"[!?]{3,}",
    ],
    "Vague / Unverifiable Claims": [
        r"\bsources say\b",
        r"\bsome people (say|claim|believe)\b",
        r"\bexperts claim\b",
        r"\bthey say\b",
        r"\baccording to insiders\b",
        r"\bunnamed sources\b",
    ],
    "Emotional Manipulation": [
        r"\bfear\b",
        r"\brage\b",
        r"\boutrage\b",
        r"\bterror\b",
        r"\bpanic\b",
        r"\bdesperate\b",
        r"\bchaos\b",
    ],
}


# ─── Text Preprocessing ───────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Clean and normalize input text before BERT tokenization.
    - Strips excess whitespace
    - Removes URLs
    - Removes excessive punctuation runs
    - Normalizes unicode quotes
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Normalize quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Remove HTML tags if any
    text = re.sub(r"<[^>]+>", " ", text)

    # Collapse excessive punctuation (e.g. !!!!! → !)
    text = re.sub(r"([!?.]){3,}", r"\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── Confidence Label ─────────────────────────────────────────────────────────
def get_confidence_label(prob: float) -> Tuple[str, str]:
    """
    Map a probability to a human-readable confidence label and color.
    Returns (label, hex_color).
    """
    if prob >= 0.90:
        return "Very High", "#ff2d78"
    elif prob >= 0.75:
        return "High", "#ff6b35"
    elif prob >= 0.60:
        return "Moderate", "#ffd700"
    elif prob >= 0.50:
        return "Low", "#aaffcc"
    else:
        return "Very Low", "#00ff87"


# ─── Credibility Score ────────────────────────────────────────────────────────
def get_credibility_score(text: str, real_prob: float) -> float:
    """
    Compute a composite credibility score (0–10) combining:
    - BERT real probability (70% weight)
    - Heuristic penalty for suspicious patterns (30% weight)
    """
    # BERT component (0–10)
    bert_score = real_prob * 10

    # Heuristic: count suspicious pattern matches
    penalty = 0
    text_lower = text.lower()
    for category, patterns in SUSPICIOUS_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                penalty += 0.4  # each hit reduces score

    # Cap penalty at 3 points
    penalty = min(penalty, 3.0)

    raw = bert_score - penalty
    return round(max(0.0, min(10.0, raw)), 1)


# ─── Highlight Suspicious Words ───────────────────────────────────────────────
def highlight_suspicious_words(text: str) -> str:
    """
    Return an HTML string where suspicious words/phrases are wrapped in
    a <span class='highlight-suspicious'> for display.
    """
    # Flatten all patterns
    all_patterns = [p for patterns in SUSPICIOUS_PATTERNS.values() for p in patterns]

    def replacer(match):
        return f"<span class='highlight-suspicious'>{match.group(0)}</span>"

    for pattern in all_patterns:
        text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)

    return text


# ─── Text Statistics ──────────────────────────────────────────────────────────
def get_text_stats(text: str) -> dict:
    """Return basic statistics about the input text."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    avg_word_len = (
        sum(len(w.strip(string.punctuation)) for w in words) / len(words)
        if words else 0
    )

    caps_ratio = (
        sum(1 for c in text if c.isupper()) / len(text) if text else 0
    )

    exclamation_count = text.count("!")
    question_count = text.count("?")

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": round(avg_word_len, 2),
        "caps_ratio": round(caps_ratio * 100, 1),
        "exclamation_marks": exclamation_count,
        "question_marks": question_count,
        "char_count": len(text),
    }
