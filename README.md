# ✦ Context-Aware Spell Checker

A smart spell checker that understands **context** — powered by a Bigram N-gram language model and Levenshtein Distance. Unlike traditional spell checkers that only flag unknown words, this tool detects **real-word errors** (e.g., *"going to the **see**"*) by analyzing the probability of word sequences in context.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)

---

##  Features

- **Context-Aware Detection** — Uses bigram probabilities to spot words that are spelled correctly but used incorrectly
- **Real-Word Error Correction** — Catches homophones and confusable pairs (*see/sea*, *blew/blue*, *fare/fair*, *rode/road*)
- **Paragraph Support** — Paste entire articles or essays — handles multi-paragraph text with sentence-level context
- **Confidence Scoring** — Every correction comes with a probability-based confidence score
- **Interactive Web UI** — Clean, minimal Streamlit interface with inline highlighting, detailed tables, and side-by-side diff view
- **Zero API Calls** — Everything runs locally. No external services, no rate limits

---

##  How It Works

### Phase 1: Training — Bigram Language Model

The system downloads a corpus of public domain literature (Pride & Prejudice, Frankenstein, Shakespeare) and builds statistical models:

```
Unigram Frequency:  { "the": 52481, "sea": 142, "see": 376, ... }
Bigram Frequency:   { ("to", "the"): 1847, ("the", "sea"): 23, ("the", "see"): 0, ... }
```

Probabilities are computed with **Laplace Smoothing**:

```
P(w₂ | w₁) = (count(w₁, w₂) + 1) / (count(w₁) + V)
```

### Phase 2: Detection & Correction

Two strategies work together:

1. **Confusable Word Pairs** — A curated list of 100+ commonly confused word pairs. When one appears, the model checks bigram counts (before AND after the word) to determine if its partner fits the context better.

2. **Unknown/Rare Words** — For words with zero bigram co-occurrence and low frequency, candidates are generated via **Levenshtein Distance** (edit distance ≤ 2) and scored by bigram frequency, unigram frequency, and edit distance penalty.

### Phase 3: Confidence Scoring

Each correction reports a confidence score based on the probability increase:

```
Confidence = (P(new | prev) - P(old | prev)) / P(new | prev) × 100
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone or navigate to the project
cd spell_checker

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### First Use

1. Click **"Download Corpus"** in the sidebar (downloads ~6.7 MB of public domain text)
2. Paste or type your text in the input area
3. Click **"✦ Correct"** to see results

---

##  Project Structure

```
spell_checker/
├── spell_checker.py      # Core engine (bigram model + correction logic)
├── corpus_downloader.py  # Corpus download & caching (Project Gutenberg)
├── app.py                # Streamlit web interface
├── requirements.txt      # Python dependencies
├── corpus/               # Cached training corpus (auto-created)
│   └── training.txt
└── README.md
```

---

## Example Corrections

| Input | Correction | Confidence |
|-------|-----------|------------|
| *I am going to the **see*** | *sea* | 98.68% |
| *She has a **blew** dress* | *blue* | 75.00% |
| *The weather is **fare** today* | *fair* | 95.24% |
| *Down the **rode** he went* | *road* | 94.12% |
| *It was a **grate** idea* | *great* | 99.44% |

---

##  Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| **Edit Distance** | 2 | Maximum Levenshtein distance for candidate generation (1 or 2) |
| **Probability Threshold** | 0.0001 | Bigram probability threshold below which a word is flagged |

---

##  Run Tests

```bash
python spell_checker.py
```

This downloads the corpus (if not cached), trains the model, and runs 5 real-word error test cases plus a paragraph-level test.

---

##  Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                   │
│  ┌───────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  Text     │  │  Inline      │  │  Table   │  │
│  │  Input    │→ │  Highlighted │  │  + Chart │  │
│  └───────────┘  └──────────────┘  └──────────┘  │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│             BigramSpellChecker                  │
│  ┌─────────────────┐  ┌──────────────────────┐  │
│  │  Bigram Model   │  │  Confusable Pairs    │  │
│  │  (Laplace)      │  │  (100+ word pairs)   │  │
│  └────────┬────────┘  └──────────┬───────────┘  │
│           │                      │              │
│  ┌────────▼──────────────────────▼───────────┐  │
│  │         Correction Engine                 │  │
│  │  • Context scoring (prev + next bigrams)  │  │
│  │  • Levenshtein candidate generation       │  │
│  │  • Confidence scoring                     │  │
│  └───────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Training Corpus                    │
│  • Pride and Prejudice (Austen)                 │
│  • Frankenstein (Shelley)                       │
│  • The Complete Works of Shakespeare            │
└─────────────────────────────────────────────────┘
```

---

##  Dependencies

| Package | Purpose |
|---------|---------|
| **Levenshtein** | Fast edit distance computation (C-backed) |
| **Streamlit** | Interactive web UI framework |
| **Requests** | Corpus download from Project Gutenberg |

---

##  Limitations

- **Corpus-dependent** — Corrections are only as good as the training data. Words that appear frequently in the corpus but rarely in modern usage may not be flagged.
- **Sentence-level context** — Bigram context resets at sentence boundaries. Cross-sentence errors are not detected.
- **Language** — English only.
- **Confusable pairs** — Relies on a curated list. Unlisted confusable pairs won't be detected.

---

##  License

MIT License — feel free to use, modify, and distribute.

---

<p align="center">Built with ❤️ using Bigram Models & Levenshtein Distance</p>
