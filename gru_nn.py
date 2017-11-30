import torch
import torch.nn as nn

class RNN_GRU(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_size, output_size, batch_size):
        super(RNN_GRU, self).__init__()
        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(output_size)
        self.batch_size = batch_size

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(self.batch_size, len(sentence), -1)
        h_gru = self.gru(x)
        o_linear = self.linear(h_gru)
        y_predict = self.softmax(o_linear)
        return y_predict

# batch_size, input_size, hidden_size, output_size = 32, 50, 200, 20
#
# x = Variable(torch.randn(batch_size, input_size))
# y = Variable(torch.randn(batch_size, output_size), requires_grad=True)
#
# model = RNN_GRU(input_size, hidden_size, output_size)
#
# criterion = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-4)
# epoch_num = 500
#
# for t in range(epoch_num):
#     y_pred = model(x)
#
#     loss = criterion(y_pred, y)
#     print(t, loss.data[0])
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()