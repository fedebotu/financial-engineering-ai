import torch
import torch.nn as nn
from .models import *
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support


class LitMLP(pl.LightningModule):
    '''
    MLP model based on Pytorch Lightning
    Made for classification and useful for imbalanced datasets
    '''
    def __init__(self, indim, outdim, 
                 hdim = 256,
                 loss = nn.BCEWithLogitsLoss(),
                 lr=1e-3,
                 weight_decay=1e-5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(indim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, outdim))   
        # this performs best in classification tasks
        # weight is used for the class imbalances
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def predict(self, x):
        # This predicts the probability distribution of the outputs
        out = torch.sigmoid(self(x))
        return out
    
    def predict_binary(self, x):
        # This predicts the label by rounding for the output
        out = torch.round(torch.sigmoid(self(x)))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out.view(-1), y)        
        self.log('train_loss', loss)
        return loss
    
    def validation_step():
        out = self(test_x)
        val_loss = self.loss(out.view(-1), test_y)
        self.log('val_loss', val_loss)
        return val_loss
    
#     def training_epoch_end(self, outputs):
#         p, r, f, _ = precision_recall_fscore_support(y_test, torch.round(self.predict(test_x)).detach().numpy())
#         self.log('precision_0', p[0])
#         self.log('precision_1', p[1])
#         self.log('recall_0', r[0])
#         self.log('recall_1', r[1])
#         self.log('f1_score_0', f[0])
#         self.log('f1_score_1', f[1])
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)       
    

class LitTimeSeriesModule(pl.LightningModule):
    def __init__(self, model_type='LSTM',
                num_classes=1, 
                input_size=10,
                hidden_size=1, 
                num_layers=1, 
                seq_length=5,
                loss = torch.nn.MSELoss(),
                lr=1e-3,
                weight_decay=1e-5):
        super().__init__()
        if model_type == 'LSTM':
            self.model = LSTM_model(num_classes, input_size, hidden_size, num_layers, seq_length)
        elif model_type == 'GRU':
            self.model = GRU_model(num_classes, input_size, hidden_size, num_layers, seq_length)
        elif model_type == 'RNN':
            self.model = RNN_model(num_classes, input_size, hidden_size, num_layers, seq_length)
        else: raise NotImplementedError('Available models: RNN, LSTM, GRU')
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
    
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer