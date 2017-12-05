import torch.nn as nn
import torch
from torch.autograd import Variable

class LSTM_Classifier(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_size, output_size, nn_layers, bidir=False):
        super(LSTM_Classifier, self).__init__()

        self.hidden_size = hidden_size
        self.nn_layers = nn_layers
        self.bidir = bidir

        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=self.bidir)
        self.linear = nn.Linear(hidden_size*(2 if self.bidir else 1), output_size)

    def init_hidden(self, batch_size):
        h_0 = Variable(torch.zeros(self.nn_layers * (2 if self.bidir else 1), batch_size, self.hidden_size))  # *2 for bidirection
        c_0 = Variable(torch.zeros(self.nn_layers * (2 if self.bidir else 1), batch_size, self.hidden_size))
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        return h_0, c_0

    def forward(self, sentences, sentences_mask):
        batch_size = sentences.data.shape[1]
        embeds = self.word_embeddings(sentences).float()
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embeds, sentences_mask)
        output, hn = self.lstm(packed_embedding, self.init_hidden(batch_size))
        if self.bidir:
            hidden = torch.cat((hn[0][0], hn[0][1]), dim=1)
            o_linear = self.linear(hidden) # this for bidir
        else:
            o_linear = self.linear(hn[0][0]) # normal (no bidir)
        return o_linear
