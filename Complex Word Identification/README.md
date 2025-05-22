
# 🔍 Complex Word Identification (CWI) for Lexical Simplification

This project implements two approaches for identifying complex words in context, which is a crucial first step in Lexical Simplification pipelines. The goal is to predict whether a target word in a sentence is complex (challenging for a reader) or simple, based on a variety of linguistic and semantic features.

---

## 📚 Datasets Used

- **CWI 2018 Shared Task**  
  - Domains: News, WikiNews, Wikipedia  
  - Labels: Binary and probabilistic complexity scores

- **MLSP 2024 Shared Task (LCP)**  
  - Trial split: Used for model training  
  - Test split: Used for final evaluation

---

## 🧠 Approaches Implemented

1. **Feature-Based MLP Classifier (PyTorch)**  
   - A multi-layer perceptron trained on handcrafted linguistic features.

2. **Soft Voting Ensemble Classifier**  
   - Combines predictions from Logistic Regression, Random Forest, and SVM.

---

## 🔍 Features Extracted

- **Surface Features**:  
  - Word length, syllable count, vowel/consonant ratio, capitalization, stopword flag

- **Lexical Features**:  
  - Zipf frequency, sentence length

- **Semantic Features**:  
  - WordNet relations

- **Psycholinguistic Features** (from MRC Database):  
  - Age of Acquisition (AoA), familiarity, concreteness, imagery

---

## 📊 Evaluation Metrics

- Accuracy  
- Macro F1-Score  
- Confusion Matrix  
- Top Misclassified Examples (Error Analysis)

---

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/CWI-LexicalSimplification.git
   cd CWI-LexicalSimplification
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook Complex_Word_Identification_(1).ipynb
   ```

---

## 🔬 Objective

To identify whether a given word in context is **complex or simple**, supporting downstream Lexical Simplification tasks such as substitution generation and sentence rephrasing.

---

