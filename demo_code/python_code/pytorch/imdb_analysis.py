# imdb_analysis
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
import random

SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

# hyper parameter
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# data load
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=True, batch_first=True)

train_set, test_set = datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train_set, min_freq=5)
LABEL.build_vocab(train_set)

vocab_size = len(TEXT.vocab) # 단어 집합의 크기
n_classes = 2 # 클래스의 개수

train_set, val_set = train_set.split(split_ratio=0.8)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_set, val_set, test_set), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)

# RNN model
class GRU(nn.Module):
        def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
                super(GRU, self).__init__()
                self.n_layers = n_layers
                self.hidden_dim = hidden_dim

                self.embed = nn.Embedding(n_vocab, embed_dim)
                self.dropout = nn.Dropout(dropout_p)
                self.gru = nn.GRU(embed_dim, self.hidden_dim,
                                   num_layers=self.n_layers, batch_first=True)
                self.out = nn.Linear(self.hidden_dim, n_classes)

        def forward(self, x):
                x = self.embed(x)
                h_0 = self._init_state(batch_size=x.size(0))
                x, _ = self.gru(x, h_0)
                h_t = x[:, -1, :]

                self.dropout(h_t)
                logit = self.out(h_t)
                return logit

        def _init_state(self, batch_size=1):
                weight = next(self.parameters()).data
                return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# model training
def train(model, optimizer, train_iter):
        model.train()
        for b, batch in enumerate(train_iter):
                x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
                y.data.sub_(1)
                optimizer.zero_grad()

                logit = model(x)
                loss = F.cross_entropy(logit, y)
                loss.backward()
                optimizer.step()

# model evaluate
def evaluate(model, val_iter):
        model.eval()
        corrects, total_loss = 0, 0
        for batch in val_iter:
                x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
                y.data.sub_(1)
                logit = model(x)
                loss = F.cross_entropy(logit, y, reduction='sum')
                total_loss += loss.item()
                corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

        size = len(val_iter.dataset)
        avg_loss = total_loss / size # 손실
        avg_accuracy = 100.0 * corrects / size # 정확도

        return avg_loss, avg_accuracy

# main
best_val_loss = None
for e in range(1, EPOCHS + 1):
        train(model, optimizer, train_iter)
        val_loss, val_accuracy = evaluate(model, val_iter)
        #print('loss : {0}, accuracy : {1}'.format(e, val_loss, val_accuracy))




