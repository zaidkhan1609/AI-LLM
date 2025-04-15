## **🧠 Fine-Tuning BERTweet for Multi-Class Sentiment Classification**
This project demonstrates how to fine-tune vinai/bertweet-base, a BERT-based transformer pretrained on Twitter data, for a 5-class sentiment classification task. It includes data preprocessing, model training, evaluation, and saving the best-performing model.

## **🚀 Features**
Fine-tunes bertweet-base using Hugging Face Transformers & PyTorch

Prepares tweet–target pairs for stance detection or sentiment classification

Implements custom training loop with learning rate scheduling and gradient clipping

Evaluates using macro-averaged F1 score and saves best checkpoint

Works on GPU or CPU automatically

## **📁 Dataset**
Uses 3 CSV files loaded directly from GitHub:

train.csv, dev.csv, test.csv
Each contains text, target, and gold_label columns.



## **📦 Dependencies**
Install dependencies in a Python environment (keep versions consistent):


pip install pandas==1.3.5 transformers==4.46.3 scikit-learn==1.0.2 numpy==1.19.5
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

python 2ndmodel.py
## **🔍 Evaluation**
Prints performance after each epoch on the dev set

Selects and saves the model with the best macro F1 score (model.pt)

Loads the best checkpoint and evaluates on the test set

## **📊 Output**
You'll see:

Epoch-wise training loss

Classification report (precision, recall, F1) for dev and test sets

## **📂 Structure**

2ndmodel.py           # Main training & evaluation script
model.pt              # Saved best model (after training completes)
## **💡 Future Improvements**
Add early stopping and WandB/MLflow tracking

Try different transformers like RoBERTa or DeBERTa

Experiment with class weighting or data augmentation
