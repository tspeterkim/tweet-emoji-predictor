import numpy as np
import utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torch
from torch.autograd import Variable


#want to try out adaptive optimizations

run_LSTM = True
run_BD_LSTM = False

learning_rate = 0.001 #arbitrary rn

class LSTM(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)


		#word embedding is input, outputs are hidden states (dim = hidden_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)


		#need to map hidden state space to output space w/ linear transform
		self.hidden2output = nn.Linear(hidden_dim, output_size)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		#need to start hidden state variables ?
		return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
	        	autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
		out_linear = self.hidden2output(lstm_out.view(len(sentence), -1))

		#output softmax
		y_predict = func.log_softmax(out_linear)
		return y_predict




class BD_LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(BD_LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

		#multiply hidden size by 2 for second layer
		self.fc = nn.Linear(hidden_size*2, num_classes)  

	def forward(self, x):
		# Set initial states
		h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection 
		c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))

		# Forward propagate RNN
		y_predict, _ = self.lstm(x, (h0, c0)) 

		# Decode hidden state of last time step
		y_predict = self.fc(y_predict[:, -1, :])  
		y_pred_soft = func.log_softmax(y_predict)
		return y_pred_soft







if __name__ == '__main__':
	main()