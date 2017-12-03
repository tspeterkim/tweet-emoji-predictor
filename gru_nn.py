import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class GRU_Classifier(nn.Module):
    # 1. zero padding use pack_padded to adjust
    # 2. learning rate and back propagation function selection
    # 3. use of last state of GRU result

    def __init__(self, vocabulary_size, embedding_dim, hidden_size, output_size, nn_layers):
        super(GRU_Classifier, self).__init__()
        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.nn_layers = nn_layers
        # self.word_embeddings.weight.data.uniform_(-0.1, 0.1)

    def forward(self, sentences, sentences_mask):
        batch_size = sentences.data.shape[1]
        embeds = self.word_embeddings(sentences).float()
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embeds, sentences_mask)
        _, h_gru = self.gru(packed_embedding, self.init_hidden(batch_size))
        o_linear = self.linear(h_gru[0,:,:])
        # o_linear = self.linear(h_gru[:,-1,:])
        return o_linear

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.nn_layers, batch_size, self.hidden_size))
