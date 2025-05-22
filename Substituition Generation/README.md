
# 🧠 Lexical Substitution Generation for MLSP 2024 Shared Task

This project implements and compares two approaches for **Lexical Substitution Generation (SG)**. The aim is to suggest simpler substitutes for complex words while preserving sentence meaning and grammatical correctness.

---

## 🚀 Goal

To support the Lexical Simplification pipeline by generating simpler word alternatives for identified complex words in context, targeting applications like language accessibility, education, and assistive tech.

---

## 📂 Dataset

- **MLSP 2024 LS Shared Task**
  - Contains complex words in context along with gold-standard substitution candidates.
  - Evaluation focuses on the ability to match these gold substitutions.

---

## 🔍 Methods Implemented

### 1. **BERT-based Masked Language Modeling (MLM)**
- The target word is masked using `[MASK]` in the sentence.
- Substitutes are generated from a fine-tuned BERT (`bert-base-uncased`) model.
- Outputs Top-5 predictions per instance.

### 2. **Multi-Source Dictionary with Zipf Filtering**
- Retrieves substitute candidates from:
  - [DictionaryAPI.dev](https://dictionaryapi.dev/)
  - Wiktionary
  - WordNet
- Filters candidates based on:
  - POS tagging
  - Word simplicity using **Zipf frequency**
- Ranks candidates by **simplicity** and **semantic similarity** (using SBERT).

---

## 📊 Evaluation Metrics

- **Potential**: % of instances with at least one correct candidate.
- **Precision / Recall / F1-Score**: Match quality against gold-standard substitutions.
- **Potential@1**: Measures quality of top-1 suggestion.

---

## 🧪 Results & Analysis

- Quantitative metrics are reported for both methods.
- Includes qualitative **error analysis** highlighting strengths and weaknesses.
- BERT tends to generate fluent, contextual substitutes.
- Zipf-based filtering improves simplicity but may miss nuanced meanings.

---

## 🛠️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Lexical-Substitution-MLSP2024.git
   cd Lexical-Substitution-MLSP2024
   ```

2. Install dependencies:
   ```bash
   pip install transformers sentence-transformers nltk spacy wordfreq requests
   python -m nltk.downloader wordnet
   python -m spacy download en_core_web_sm
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Substituition_generation_MLSP2024.ipynb
   ```

---

## ✅ Highlights

- Hybrid approach combining neural models with curated linguistic resources.
- Supports evaluation with custom gold annotations.
- Easy to extend to multilingual or domain-specific settings.

---
