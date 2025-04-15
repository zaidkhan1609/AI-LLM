#Importing Libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np

#Using the GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Function for loading data sets
def load_data():
    train_df = pd.read_csv('https://raw.githubusercontent.com/osamuzahid/NLP/refs/heads/main/train.csv')
    dev_df = pd.read_csv('https://raw.githubusercontent.com/osamuzahid/NLP/refs/heads/main/dev.csv')
    test_df = pd.read_csv('https://raw.githubusercontent.com/osamuzahid/NLP/refs/heads/main/test.csv')
    return train_df, dev_df, test_df

#Function for preprocessing and processing data
def preprocess_data(texts, targets, labels, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    
    for text, target in zip(texts, targets):
        combined_text = f"{text} {tokenizer.sep_token} {target}"
        encoding = tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_masks, labels)

#Function for creating Dataloaders
def create_dataloaders(train_ds, dev_ds, test_ds, batch_size=16):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, dev_loader, test_loader

#Function for training epochs
def train_epoch(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(data_loader)

#Function for evaluating the Model
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return classification_report(actual_labels, predictions, digits=4)

#Function for initializing the tokenizer and model
def initialize_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/bertweet-base",
        num_labels=5  # Adjusted per the number of classes in the data set
    ).to(device)
    return tokenizer, model

#Defining the Optimizer and Scheduler
def get_optimizer_and_scheduler(model, train_loader, num_epochs, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler

#Function for training and evaluating
def train_and_evaluate():
    train_df, dev_df, test_df = load_data()
    tokenizer, model = initialize_model_and_tokenizer()
    
    # Create datasets
    train_ds = preprocess_data(
        train_df['text'].values,
        train_df['target'].values,
        train_df['gold_label'].values,
        tokenizer
    )
    dev_ds = preprocess_data(
        dev_df['text'].values,
        dev_df['target'].values,
        dev_df['gold_label'].values,
        tokenizer
    )
    test_ds = preprocess_data(
        test_df['text'].values,
        test_df['target'].values,
        test_df['gold_label'].values,
        tokenizer
    )
    
    # Create dataloaders
    train_loader, dev_loader, test_loader = create_dataloaders(train_ds, dev_ds, test_ds)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, train_loader, num_epochs=7)
    
    # Training loop
    best_f1 = 0
    for epoch in range(7):
        print(f"Epoch {epoch + 1}/7")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Average train loss: {train_loss:.4f}")
        
        dev_report = evaluate(model, dev_loader, device)
        print("\nDev Set Results:")
        print(dev_report)
        
        # Save the best model
        dev_f1 = float(dev_report.split('\n')[-2].split()[-2])  # Get macro-averaged F1
        if dev_f1 > best_f1: #Compare macro F1 to save the best model state after each epoch
            best_f1 = dev_f1
            torch.save(model.state_dict(), 'model.pt')
    
    # Load best model and test
    model.load_state_dict(torch.load('model.pt'))
    test_report = evaluate(model, test_loader, device)
    print("\nTest Set Results:")
    print(test_report)

#Execute Training and Evaluation
train_and_evaluate()
