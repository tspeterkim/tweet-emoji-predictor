import torch
import torch.nn as nn
from torch.autograd import Variable

class GRU_Classifier(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_size, output_size, batch_size):
        super(GRU_Classifier, self).__init__()
        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.hidden = self.init_hidden(hidden_size)

    def forward(self, sentences):
        embeds = self.word_embeddings(sentences)
        x = embeds.view(len(sentences), -1, self.embedding_dim)
        h_gru, self.hidden = self.gru(x, self.hidden)
        o_linear = self.linear(h_gru)
        y_predict = self.softmax(o_linear)
        return y_predict

    def init_hidden(self, hidden_size):
        return Variable(torch.randn(1, self.batch_size, hidden_size))
