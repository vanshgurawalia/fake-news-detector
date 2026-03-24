# 🛡️ FakeShield — Fake News Detector

> Can AI tell the difference between real news and complete nonsense? Turns out, yes — most of the time. This project uses BERT to classify news articles and headlines as real or fake, wrapped in a clean Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![BERT](https://img.shields.io/badge/Model-BERT-orange?style=flat-square)

---

## 🤔 Why I Built This

Fake news is everywhere and it's genuinely hard to spot. I wanted to see how well a transformer model like BERT could handle it — and more importantly, build something visual and interactive around it rather than just a notebook. This project is the result of that curiosity.

---

## 🎬 What It Does

You paste in any news headline or article, hit Analyze, and the app tells you:

- ✅ Whether it's **REAL or FAKE**
- 📊 The exact probability scores from BERT
- 🎯 A credibility score out of 10
- 🚩 Which suspicious patterns it found (clickbait, conspiracy language, sensationalism, etc.)
- 🔤 Optional token breakdown if you're curious how BERT sees the text

It's not perfect — no model is — but it's surprisingly good at catching the obvious stuff.

---

## 🧠 How It Works

The core idea is pretty straightforward:

```
Your text
   ↓
Clean & preprocess
   ↓
BERT tokenizer (WordPiece)
   ↓
bert-base-uncased (110M parameters)
   ↓
[CLS] token → classification head
   ↓
Softmax → P(REAL) and P(FAKE)
```

BERT reads the entire sentence at once (not word by word), so it understands context really well. The `[CLS]` token at the start acts as a summary of the whole input, which gets fed into a simple linear layer to make the final call.

---

## 🚀 Running It Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/fakeshield.git
cd fakeshield
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python -m streamlit run app.py
```

It'll download the model on first run (~400MB). After that it's instant.

> **Windows users:** if `streamlit` isn't recognized, use `python -m streamlit run app.py`

---

## 📁 Project Structure

```
fakeshield/
├── app.py            ← the entire Streamlit UI lives here
├── train.py          ← fine-tune BERT on your own dataset
├── predict.py        ← run predictions from the command line
├── utils.py          ← text cleaning, pattern detection, scoring
├── requirements.txt
└── README.md
```

---

## 🏋️ Want to Fine-Tune on Your Own Data?

The app uses a pre-trained model by default. If you want to train from scratch on your own dataset, you'll need a CSV with two columns — `text` and `label` (0 for real, 1 for fake).

Good free datasets to start with:
- **WELFake** — 72K articles, easy to find on Kaggle
- **ISOT** — 44K articles from University of Victoria
- **LIAR** — 12K statements with fact-check labels

Then just run:
```bash
python train.py --data_path yourdata.csv --output_dir saved_model/ --epochs 3
```

And point the app to your model by changing one line in `app.py`:
```python
model_name = "./saved_model"
```

Expected accuracy after fine-tuning: around **97–99%** on ISOT, slightly lower on harder datasets like LIAR.

---

## 📊 App Screenshots

> *(add your own screenshots here — people really do judge projects by how they look)*

---

## ⚠️ Honest Limitations

I want to be upfront about what this can and can't do:

- The **base model** (without fine-tuning) is essentially guessing. Fine-tune it for real results.
- It struggles with **satire and opinion pieces** — these are genuinely hard even for humans.
- Short headlines give the model very little to work with. Longer text = more accurate.
- This is a **demo project**, not a production misinformation detector. Don't use it to make real-world judgments about news sources.

---

## 🛠️ Tech Stack

| Tool | Why |
|------|-----|
| BERT | Transformer model that understands context bidirectionally |
| HuggingFace Transformers | Easy model loading and tokenization |
| Streamlit | Fast way to build ML demos without writing frontend code |
| Plotly | Interactive charts that actually look nice |
| PyTorch | Backend for model inference |

---

## 🤝 Contributing

Found a bug? Have an idea? PRs are welcome. A few things that would make this better:

- Adding explainability (attention visualization would be cool)
- Supporting more languages with multilingual BERT
- A FastAPI backend so it can be used as an API
- Better handling of satire and opinion content

---

## 📄 License

MIT — do whatever you want with it, just don't use it to spread more fake news 😄

---

*Built with curiosity and too many cups of chai ☕*
