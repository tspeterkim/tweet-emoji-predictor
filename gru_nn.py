import torch
import torch.nn as nn
import torch.autograd.variable as variable
import torch.optim as optim

class RNN_GRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(output_size)

    def forward(self, x):
        h_gru = self.gru(x)
        o_linear = self.linear(h_gru)
        y_predict = self.softmax(o_linear)
        return y_predict

batch_size, input_size, hidden_size, output_size = 32, 50, 200, 20

x = variable(torch.randn(batch_size, input_size))
y = variable(torch.randn(batch_size, output_size), requires_grad=True)

model = RNN_GRU(input_size, hidden_size, output_size)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)
epoch_num = 500

for t in range(epoch_num):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()