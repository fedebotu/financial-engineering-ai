import torch
from torch import nn
from torch.autograd import Variable 

class RNN_model(nn.Module):
    '''Recurrent Neural Network model'''
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(RNN_model, self).__init__()
        self.RNN_model = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #RNN layer
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer

    def forward(self,x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x) #hidden state
        # Propagate input through RNN
        out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out

class GRU_model(nn.Module):
    '''Gated Recurrent Unit model'''
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU_model, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #GRU layer
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer

    def forward(self,x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x) #hidden state
        # Propagate input through GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out

    
class LSTM_model(nn.Module):
    '''Long Short Term Memory model'''
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_model, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #LSTM layer
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer

    def forward(self,x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x) #hidden state
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x) #internal state   
        # Propagate input through LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out


