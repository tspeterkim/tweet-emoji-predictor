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
	def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, batch_size):
		super(LSTM, self).__init__()

		self.hidden_size = hidden_size
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)


		#word embedding is input, outputs are hidden states (dim = hidden_size)
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

		self.logsoftmax = nn.LogSoftmax()

		#need to map hidden state space to output space w/ linear transform
		self.hidden2output = nn.Linear(hidden_size, output_size)
		self.hidden = self.init_hidden(hidden_size, batch_size)

	def init_hidden(self, hidden_size, batch_size):
		#need to start hidden state variables ?
#		return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
#	        	autograd.Variable(torch.zeros(1, 1, self.hidden_size)))

		return Variable(torch.randn(1,batch_size, hidden_size)), Variable(torch.randn(1, batch_size, hidden_size))

		#return Variable(torch.randn(1, batch_size, hidden_size))

	def forward(self, sentence, batch_size):
		#remove this
		self.hidden = self.init_hidden(self.hidden_size, batch_size)

		embeds = self.word_embeddings(sentence)
		lstm_out, self.hidden = self.lstm(embeds, self.hidden)#self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
		#out_linear = self.hidden2output(lstm_out.view(len(sentence), -1))
		out_linear = self.hidden2output(lstm_out[:,-1,:])

		#output softmax
		y_predict_log = self.logsoftmax(out_linear)

		y_predict = func.log_softmax(out_linear)
		return y_predict_log, y_predict




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




