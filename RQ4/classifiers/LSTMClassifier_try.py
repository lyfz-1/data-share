import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import time

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)

class ModelConfig:
    batch_size = 32
    output_size = 2
    hidden_dim = 384
    n_layers = 2
    lr = 2e-5
    bidirectional = False  # LSTM
    drop_prob = 0.55
    # training params
    epochs = 10
    print_every = 10
    clip = 5  # gradient clipping
    use_cuda = USE_CUDA
    codebert_path = 'your codebert path'
    save_path = 'best_classifier.pth'
    sampleRate = 2

class bert_lstm(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(bert_lstm, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.bert = BertModel.from_pretrained(bertpath)

        for param in self.bert.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        self.dropout = nn.Dropout(drop_prob)

        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        # self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = self.bert(x)[0]
        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)

        if self.bidirectional:
            hidden_last_L = hidden_last[-2]
            hidden_last_R = hidden_last[-1]
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
        else:
            hidden_last_out = hidden_last[-1]

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 1
        if self.bidirectional:
            number = 2
        if (USE_CUDA):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                      )
        return hidden

class codebert_lstm(nn.Module):
    def __init__(self, codebert_path, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(codebert_lstm, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")

        for param in self.codebert.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        # self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = self.codebert(x)[0]
        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)

        if self.bidirectional:
            hidden_last_L = hidden_last[-2]
            hidden_last_R = hidden_last[-1]
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
        else:
            hidden_last_out = hidden_last[-1]

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 1
        if self.bidirectional:
            number = 2
        if (USE_CUDA):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                      )
        return hidden

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, output, target):
        # convert output to pseudo probability
        out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
        probs = torch.sigmoid(out_target)
        focal_weight = torch.pow(1 - probs, self.gamma)

        # add focal weight to cross entropy
        ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            focal_loss = (focal_loss / focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss


def train_model(config, data_train):

    net = codebert_lstm(config.codebert_path,
                    config.hidden_dim,
                    config.output_size,
                    config.n_layers,
                    config.bidirectional,
                    config.drop_prob
                    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    if (config.use_cuda):
        net.cuda()

    net.train()

    for e in range(config.epochs):
        # initialize hidden state
        h = net.init_hidden(config.batch_size)
        counter = 0
        # batch loop
        for inputs, labels in data_train:
            counter += 1
            if (config.use_cuda):
                inputs, labels = inputs.cuda(), labels.cuda()
            h = tuple([each.data for each in h])

            net.zero_grad()
            output = net(inputs, h)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            optimizer.step()

            # loss stats
            if counter % config.print_every == 0:
                net.eval()
                with torch.no_grad():
                    val_h = net.init_hidden(config.batch_size)
                    val_losses = []

                net.train()
                print("Epoch: {}/{}, ".format(e + 1, config.epochs),
                      "Step: {}, ".format(counter),
                      "Loss: {:.6f}, ".format(loss.item()))

    torch.save(net.state_dict(), config.save_path)


def test_model(config, data_test):
    net = codebert_lstm(config.codebert_path,
                    config.hidden_dim,
                    config.output_size,
                    config.n_layers,
                    config.bidirectional,
                    config.drop_prob
                    )

    net.load_state_dict(torch.load(config.save_path))

    if (config.use_cuda):
        net.cuda()

    net.train()
    criterion = nn.CrossEntropyLoss()
    test_losses = []  # track loss
    num_correct = 0
    net.eval()
    correctT = 0
    total = 0
    classnum = 2
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))

    # init hidden state
    h = net.init_hidden(config.batch_size)
    net.eval()

    for inputs, labels in data_test:
        h = tuple([each.data for each in h])
        if (USE_CUDA):
            inputs, labels = inputs.cuda(), labels.cuda()

        output = net(inputs, h)
        test_loss = criterion(output.squeeze(), labels.long())
        test_losses.append(test_loss.item())
        # output = torch.nn.Softmax(dim=1)(output)
        _, pred = torch.max(output, 1)

        labels = Variable(labels)

        total += labels.size(0)
        correctT += pred.eq(labels.data).cpu().sum()
        pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(output.size()).scatter_(1, labels.data.cpu().view(-1, 1).long(), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask * tar_mask
        acc_num += acc_mask.sum(0)

    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = acc_num.sum(1) / target_num.sum(1)
    recall = (recall.numpy()[0] * 100).round(3)
    precision = (precision.numpy()[0] * 100).round(3)
    F1 = (F1.numpy()[0] * 100).round(3)
    accuracy = (accuracy.numpy()[0] * 100).round(3)
    print('predict_num', " ".join('%s' % id for id in predict_num))
    print('recall', " ".join('%s' % id for id in recall))
    print('precision', " ".join('%s' % id for id in precision))
    print('F1', " ".join('%s' % id for id in F1))
    print('accuracy', accuracy)

    test_acc = num_correct / len(data_test.dataset)
    return test_acc, test_losses, recall, precision, F1, accuracy

def myDataProcess(dataFile):
    df = pd.read_csv(str(dataFile), encoding='utf-8')

    valid_num = len(df['label'].tolist())
    print(f'The number of samples:{valid_num}')

    label_counts = df['label'].value_counts()
    print(label_counts)

    df["message"].apply(lambda x: x.replace('<enter>', '$enter').replace('<tab>', '$tab'). \
                                    replace('<url>', '$url').replace('<version>', '$version') \
                                    .replace('<pr_link>', '$pull request>').replace('<issue_link >',
                                                                                    '$issue') \
                                    .replace('<otherCommit_link>', '$other commit').replace("<method_name>",
                                                                                            "$method") \
                                    .replace("<file_name>", "$file").replace("<iden>", "$token"))

    print("load data successfully!")

    diffs = list(df['diff'].array)
    labels = np.array(df['label'])

    return diffs , labels


if __name__ == '__main__':
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_config = ModelConfig()

    diffs, labels = myDataProcess("data_with_codebert_embeddings.csv")
    result_comments = diffs

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    result_comments_id = tokenizer(result_comments,padding=True,truncation=True,max_length=200,return_tensors='pt')

    X = result_comments_id['input_ids']
    y = torch.from_numpy(labels).float()

    fold = KFold(n_splits=10, random_state=6666, shuffle=True)

    PRECISION = np.array([0.0, 0.0])
    RECALl = np.array([0.0, 0.0])
    F1 = np.array([0.0, 0.0])
    ACC = 0.0

    fold_num = 1
    WeightedPRECISION = 0.0
    WeightedRECALl = 0.0
    WeightedF1 = 0.0

    for train_index, test_index in fold.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        posNum = np.sum(labels == 1)
        negNum = (int)(posNum / model_config.sampleRate)

        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_data,
                                  shuffle=True,
                                  batch_size=model_config.batch_size,
                                  drop_last=True)

        test_loader = DataLoader(test_data,
                                 shuffle=True,
                                 batch_size=model_config.batch_size,
                                 drop_last=True)
        if (USE_CUDA):
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')

        train_model(model_config, train_loader)

        label_num = y_test.numpy().tolist()
        dict = {}
        for key in label_num:
            dict[key] = dict.get(key, 0) + 1
        weighted = [dict.get(0, 0) / len(label_num), dict.get(1, 0) / len(label_num)]

        def weightedMetric(weighted, score):
            return weighted[0] * score[0] + weighted[1] * score[1]

        test_acc, test_losses, recall, precision, f1, accuracy = test_model(model_config, test_loader)

        wprecision = weightedMetric(weighted, precision)
        wrecall = weightedMetric(weighted, recall)
        wf1 = weightedMetric(weighted, f1)

        PRECISION += precision
        RECALl += recall
        F1 += f1
        WeightedPRECISION += wprecision
        WeightedRECALl += wrecall
        ACC += accuracy
        WeightedF1 += wf1

        print('Total Weighted Recall', WeightedRECALl / fold_num)
        print('Total Weighted Precision', WeightedPRECISION / fold_num)
        print('Total Weighted F1', WeightedF1 / fold_num)
        print('Total Accuracy', (ACC) / fold_num)
        print('Total Recall', RECALl / fold_num)
        print('Total Precision', PRECISION / fold_num)
        print('Total F1', F1 / fold_num)
        fold_num += 1

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Epochs:{model_config.epochs}，Time cost:{total_time}s')