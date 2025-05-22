## 🔍 Lightweight Transformer Fine-Tuning: A Comparative Study of BitFit, LoRA, and Diff Pruning
This repository contains a comprehensive evaluation of lightweight fine-tuning methods for transformer models across several benchmark NLP tasks. Specifically, it benchmarks BitFit, LoRA, Diff Pruning, and Full Fine-Tuning using models such as BERT, GPT-2, and T5 on tasks like MRPC, RTE, SST-2, and CoNLL-2003.

📁 Project Structure

📦 BitFit_Comparison_Project
├── 📄 C24081994_C24082638.ipynb       # Main notebook with training and evaluation
├── 📄 C24081994_C24082638.pdf         # Project report (PDF)
├── 📂 C240881994_C24082638/           # Data and experiment results
│   ├── conll2003_train.csv
│   ├── mrpc_train.csv
│   ├── rte_train.csv
│   └── ... (more CSV files for all datasets used)
📊 Datasets
MRPC: Microsoft Research Paraphrase Corpus

RTE: Recognizing Textual Entailment

SST-2: Stanford Sentiment Treebank

CoNLL-2003: Named Entity Recognition

🧪 Methods Compared
Method	Description
BitFit	Fine-tunes only the bias terms in transformer layers
LoRA	Injects trainable low-rank matrices into attention modules
Diff Pruning	Learns sparse parameter updates on top of frozen base model weights
Full FT	Fine-tunes all model parameters (baseline for comparison)

⚙️ Models Used
BERT-base

GPT-2 (small)

T5-small

📈 Evaluation Metrics
Accuracy

F1 Score

Precision/Recall (task-dependent)

💡 Key Findings
BitFit offers comparable performance to full fine-tuning on several tasks with significantly fewer trainable parameters.

LoRA consistently performs well with a better trade-off between performance and efficiency.

Diff Pruning yields mixed results and requires careful tuning per task.

🚀 How to Run
Install required dependencies:

bash
Copy
Edit
pip install transformers datasets sklearn
Open the Jupyter notebook:

bash
Copy
Edit
jupyter notebook C24081994_C24082638.ipynb
Follow the instructions in the notebook to run experiments or visualize results.

📌 TODOs
 Add training logs and result plots for each method

 Package the scripts for CLI-based evaluation

 Extend experiments to larger models (e.g., BERT-large, T5-base)
