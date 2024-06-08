
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import logging
logging.set_verbosity_error()
import pandas as pd
import numpy as np
from torchtext import data
import torch.nn.functional as F
import re
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import os
from torch.nn import TransformerDecoder, TransformerDecoderLayer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

class BERT2(nn.Module):
    def __init__(self, freeze_bert=False):
        super().__init__()

        config = AutoConfig.from_pretrained('microsoft/codebert-base')
        config.update({'output_hidden_states':True})
        self.bert  = AutoModel.from_pretrained('microsoft/codebert-base',config=config)

        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 2)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        out = outputs[0].mean(dim=1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def binary_accuracy(pred, y):
    pred = pred.max(1, keepdim=True)[1].squeeze(1)
    correct=(pred==y).float()
    return correct.sum()/len(correct)

def train(model,iterator,optimizer,criterion):
    model.train()
    epoch_loss=0
    epoch_acc=0
    for batch in iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        predictions = model(input_ids, attention_mask=attention_mask)
        loss=criterion(predictions,labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        epoch_acc+=binary_accuracy(predictions,labels)
        epoch_loss+=loss.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model,iterator,criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            predictions = model(input_ids, attention_mask=attention_mask)
            loss = criterion(predictions, labels)

            epoch_acc += binary_accuracy(predictions, labels)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

df = pd.read_csv('data_with_codebert_embeddings.csv',encoding='utf-8')
df.info()

x = df['diff'].tolist()
y = df['label'].apply(lambda x: int(x)).values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=128)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=128)

train_dataset = NewsDataset(train_encoding, y_train)
test_dataset = NewsDataset(test_encoding, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)

model = BERT2().to(device)
criterion = nn.CrossEntropyLoss().to(device)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

N_EPOCHS = 3

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time=time.time()

    train_loss, train_acc=train(model,train_loader,optimizer,criterion)
    valid_loss, valid_acc = evaluate(model, test_loader, criterion)


    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    if valid_loss<best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),'CodeBERT_MLP.pth')


model.eval()
test_loss = 0
correct_total = 0

pred_list = []
target_list = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        target_list += labels
        pred = model(input_ids, attention_mask=attention_mask)

        pred = pred.max(1, keepdim=True)[1].squeeze(1)
        pred_list += pred

        correct = (pred==labels).float()
        correct_total += correct.sum()/len(correct)

print('Accuracy:',float(correct_total/len(test_loader)))

pred_list = [int(i.to('cpu')) for i in pred_list]
target_list = [int(i.to('cpu')) for i in target_list]

print('Precision:',precision_score(target_list,pred_list))
print('Recall:',recall_score(target_list,pred_list))
print('F1:',f1_score(target_list,pred_list))

