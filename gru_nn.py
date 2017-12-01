import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class GRU_Classifier(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_size, output_size, batch_size):
        super(GRU_Classifier, self).__init__()
        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        # self.word_embeddings.weight.data.uniform_(-0.1, 0.1)

    def forward(self, sentences, batch_size):
        self.hidden = self.init_hidden(self.hidden_size, batch_size)
        embeds = self.word_embeddings(sentences).float()
        # print('embeds: ', embeds.size())
        # print('hidden: ', self.hidden.size())
        h_gru, self.hidden = self.gru(embeds, self.hidden)
        o_linear = self.linear(h_gru[:,-1,:])
        y_predict_log = self.logsoftmax(o_linear)
        y_predict = self.softmax(o_linear)
        return y_predict_log, y_predict

    def init_hidden(self, hidden_size, batch_size):
        return Variable(torch.randn(1, batch_size, hidden_size))
