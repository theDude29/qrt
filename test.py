import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

x_train = pd.read_csv('x_train.csv', index_col='ID')
y_train = pd.read_csv('y_train.csv', index_col='ID')
train = pd.concat([x_train, y_train], axis=1)
test = pd.read_csv('x_test.csv', index_col='ID')

# Define the PyTorch model with customizable hyperparameters
class NeuralNetRegressorWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout_rate):
        super(NeuralNetRegressorWithDropout, self).__init__()
        
        # Define layers with customizable hidden size and dropout
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_1)
        self.fc3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc4 = nn.Linear(hidden_size_2, hidden_size_2)
        self.fc5 = nn.Linear(hidden_size_2, hidden_size_2 // 2)
        self.fc6 = nn.Linear(hidden_size_2 // 2, hidden_size_2 // 4)
        
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)
        self.dropout_3 = nn.Dropout(p=dropout_rate)
        
        self.output = nn.Linear(hidden_size_2 // 4, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout_1(x)
        
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        x = self.dropout_2(x)
        
        x = F.relu(self.fc4(x))
        
        x = F.relu(self.fc5(x))
        x = self.dropout_3(x)
        
        x = F.relu(self.fc6(x))
                
        x = F.softmax(self.output(x), dim=1)
        return x

# Define an objective function for Optuna
def objective(trial):
    # Hyperparameter suggestions
    hidden_size_1 = trial.suggest_int('hidden_size_1', 32, 128, step=32)
    hidden_size_2 = trial.suggest_int('hidden_size_2', 16, 64, step=8)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.4, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Model initialization
    model = NeuralNetRegressorWithDropout(input_size=20, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, dropout_rate=dropout_rate)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dummy dataset for demonstration purposes
    X_train = x_train[]
    y_train = 
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(10):  # Reduce epochs for faster testing
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
    
    # Validation loss (replace with a proper validation set for real testing)
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_train)
        val_loss = criterion(val_predictions, y_train).item()
    
    return val_loss

# Run Optuna for hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)
