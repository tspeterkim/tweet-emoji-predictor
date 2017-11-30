import torch
import torch.nn as nn
from torch.autograd import Variable

class GRU_Classifier(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_size, output_size):
        super(GRU_Classifier, self).__init__()
        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        # self.batch_size = batch_size
        # self.hidden = self.init_hidden(hidden_size)

    def forward(self, sentences, batch_size):
        self.hidden = self.init_hidden(self.hidden_size, batch_size)
        embeds = self.word_embeddings(sentences).float()
        x = embeds.view(len(sentences), -1, self.embedding_dim)
        print(x)
        h_gru, self.hidden = self.gru(x, self.hidden)
        o_linear = self.linear(h_gru)
        y_predict = self.softmax(o_linear)
        return y_predict

    def init_hidden(self, hidden_size, batch_size):
        return Variable(torch.randn(1, batch_size, hidden_size))
