## **🏷️ Named Entity Recognition (NER) using DistilBERT & PyTorch**
This project fine-tunes the lightweight transformer model distilbert-base-uncased for Named Entity Recognition (NER) on the CoNLL-2003 dataset. It uses Hugging Face Transformers, Datasets, and PyTorch for end-to-end training and evaluation.

## **📚 Dataset: CoNLL-2003**
The CoNLL-2003 dataset is a benchmark dataset for NER tasks. It includes annotations for:

PER — Person

LOC — Location

ORG — Organization

MISC — Miscellaneous

Format: BIO tagging (e.g., B-LOC, I-PER, O)


from datasets import load_dataset
dataset = load_dataset("conll2003")


## **🧠 Model Architecture**
distilbert-base-uncased (from Hugging Face Transformers)

Linear classification layer on top of token embeddings

Uses attention masks to avoid padded token influence

## **⚙️ Training Details**

Component	Description
Optimizer	AdamW (from transformers.optimization)
Loss Function	nn.CrossEntropyLoss()
Epochs	Customizable (default: 3)
Batch Size	16
Evaluation	Token-level F1-score, classification report
## **🧪 How to Run**
1. Install dependencies:

pip install torch transformers datasets scikit-learn
2. Run the notebook:

jupyter notebook NER_using_Pytorch.ipynb
Or convert to .py and run as a script.

## **📊 Output**
Prints F1 score and classification report after training

Displays confusion matrix for each class

Saves trained model to ner_model.pt

## **📂 Project Structure**

NER_using_Pytorch.ipynb     # Full notebook for training + evaluation

✅ Key Features
✅ Fine-tunes DistilBERT for token-level classification

✅ Tokenizes inputs using DistilBertTokenizerFast

✅ Efficient batch training with attention masking

✅ Hugging Face datasets integration with PyTorch

## **🚀 Possible Enhancements**
Integrate wandb or MLflow for experiment tracking

Replace with bert-base-cased or roberta-base

Deploy model via FastAPI or Streamlit demo
