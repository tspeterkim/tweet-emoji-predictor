import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class GRU_Classifier(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_size, output_size, batch_size):
        super(GRU_Classifier, self).__init__()
        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden(hidden_size, batch_size)
        self.embedding_dim = embedding_dim
        # self.word_embeddings.weight.data.uniform_(-0.1, 0.1)

    def forward(self, sentences, batch_size):
        # self.hidden = self.init_hidden(self.hidden_size, batch_size)
        embeds = self.word_embeddings(sentences)
        print('embeds: ', embeds.size())
        print('hidden: ', self.hidden.size())
        h_gru, self.hidden = self.gru(embeds, self.hidden)
        y_predict = self.linear(h_gru)
        return y_predict

    def init_hidden(self, hidden_size, batch_size):
        return Variable(torch.randn(1, batch_size, hidden_size))
