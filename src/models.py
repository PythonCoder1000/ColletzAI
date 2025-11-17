import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import Dataset

class CollatzDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        input_val = int(parts[0])
                        output1 = int(parts[1])
                        output2 = int(parts[2])
                        self.data.append((input_val, output1, output2))
        
        self.inputs = np.array([x[0] for x in self.data], dtype=np.float32)
        self.outputs1 = np.array([x[1] for x in self.data], dtype=np.float32)
        self.outputs2 = np.array([x[2] for x in self.data], dtype=np.float32)
        
        self.input_mean = self.inputs.mean()
        self.input_std = self.inputs.std()
        self.inputs = (self.inputs - self.input_mean) / (self.input_std + 1e-8)
        
        self.output1_mean = self.outputs1.mean()
        self.output1_std = self.outputs1.std()
        self.outputs1 = (self.outputs1 - self.output1_mean) / (self.output1_std + 1e-8)
        
        self.output2_mean = self.outputs2.mean()
        self.output2_std = self.outputs2.std()
        self.outputs2 = (self.outputs2 - self.output2_mean) / (self.output2_std + 1e-8)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor([self.outputs1[idx], self.outputs2[idx]])


class MLP(nn.Module):
    """Multi-Layer Perceptron with batch normalization"""
    def __init__(self, input_dim=1, hidden_dims=[256, 512, 256, 128], output_dim=2, dropout=0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        # BatchNorm requires batch_size > 1 during training
        # If batch_size == 1 and in training mode, temporarily switch to eval
        if self.training and x.size(0) == 1:
            self.eval()
            out = self.network(x)
            self.train()
            return out
        return self.network(x)


class LSTM(nn.Module):
    """LSTM with bidirectional option"""
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.2, bidirectional=False):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer to convert scalar input to feature vector
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # Embed input: (batch_size, 1) -> (batch_size, 1, hidden_dim)
        x = self.embedding(x)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        
        # Apply dropout and output
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        
        return output


class HybridModel(nn.Module):
    """Hybrid MLP + LSTM model"""
    def __init__(self, input_dim=1, mlp_hidden_dims=[128, 256], lstm_hidden_dim=128, 
                 num_layers=2, output_dim=2, dropout=0.2):
        super(HybridModel, self).__init__()
        
        # MLP branch
        mlp_layers = []
        prev_dim = input_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # LSTM branch
        self.embedding = nn.Linear(input_dim, lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Combine branches
        combined_dim = prev_dim + lstm_hidden_dim
        self.fc1 = nn.Linear(combined_dim, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def _safe_forward(self, x, network):
        """Safely forward through network with BatchNorm, handling batch_size=1"""
        if self.training and x.size(0) == 1:
            self.eval()
            out = network(x)
            self.train()
            return out
        return network(x)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # MLP branch (with safe BatchNorm handling)
        mlp_out = self._safe_forward(x, self.mlp)
        
        # LSTM branch
        lstm_emb = self.embedding(x)
        lstm_emb = lstm_emb.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_emb)
        lstm_out = lstm_out[:, -1, :]
        
        # Combine
        combined = torch.cat([mlp_out, lstm_out], dim=1)
        combined = self.fc1(combined)
        # Safe BatchNorm for combined features
        if self.training and combined.size(0) == 1:
            self.eval()
            combined = self.bn(combined)
            self.train()
        else:
            combined = self.bn(combined)
        combined = nn.ReLU()(combined)
        combined = self.dropout(combined)
        output = self.fc2(combined)
        
        return output


class TransformerModel(nn.Module):
    """Transformer-based model for sequence prediction"""
    def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=3, 
                 dim_feedforward=512, output_dim=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # Embed and add positional encoding
        x = self.embedding(x)
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        x = x + self.pos_encoder
        
        # Transformer
        x = self.transformer(x)
        x = x[:, -1, :]  # Take last output
        
        x = self.dropout(x)
        output = self.fc(x)
        
        return output


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.saveCheckpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.saveCheckpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def saveCheckpoint(self, model):
        self.best_weights = copy.deepcopy(model.state_dict())

