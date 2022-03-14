# Reference | https://www.youtube.com/watch?v=EoGUlvhRYpk&t=2691s
# Seq2Seq Model Implementation

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import numpy as np
import spacy
import random

# CUDA configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenizer_german(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_english(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german  = Field(tokenize=tokenizer_german,  lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_english, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, validate_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x -> (seq_len, N)
        # embedding -> (seq_len, N, embedding_size)

        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        # x -> (N) but (1, N) (1 represents a single word)
        # embedding -> (1, N, embedding_size)
        # output -> (1, N, hidden_size)
        # predictions -> (1, N, len_vocab)

        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # starting token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output

            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        return outputs

# Hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

load_model = False
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validate_data, test_data),
    batch_size=batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src),
    device=device
)

encoder = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
decoder = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

pad_idx = english.vocab.stoi['<pad>']
loss_function = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Training 
for epoch in range(num_epochs):
    print(f"Epoch [{epoch} / {num_epochs}]")

    for batch_idx, batch in enumerate(train_iterator):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(input_data, target)
        # output[0] -> start token
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].resahpe(-1)

        optimizer.zero_grad()
        loss = loss_function(output, target)
        loss.backward()

        # avoid exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()