import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VegetationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.LongTensor(y.values)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class OptimalFeatureLearning(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 8, 4, 2]):
        super(OptimalFeatureLearning, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        # Final classification layer
        layers.append(nn.Linear(hidden_dims[-1], 2))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return x, activations
    
    def get_separating_margin(self, x, y):
        """Calculate separating margin between classes in latent space"""
        with torch.no_grad():
            _, activations = self.forward(x)
            hidden_features = activations[-2]  # Features before final layer
            
            # Separate features by class
            class_0_features = hidden_features[y == 0]
            class_1_features = hidden_features[y == 1]
            
            # Calculate pairwise distances
            distances = torch.cdist(class_0_features, class_1_features)
            
            # Get minimum distance (margin)
            margin = torch.min(distances)
            
            return margin.item()

def hinge_loss(outputs, targets, margin=1.0):
    """
    Custom hinge loss with margin maximization
    """
    batch_size = outputs.size(0)
    
    # Convert targets to {-1, 1}
    y = 2 * targets.float() - 1
    
    # Calculate hinge loss
    loss = torch.mean(torch.clamp(margin - outputs * y, min=0))
    return loss

def train_model(model, train_loader, val_loader, n_epochs=100, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = hinge_loss
    
    best_val_margin = 0
    best_model = None
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_margin = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs, _ = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                val_margin += model.get_separating_margin(batch_X, batch_y)
        
        val_margin /= len(val_loader)
        
        if val_margin > best_val_margin:
            best_val_margin = val_margin
            best_model = model.state_dict()
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, '
                  f'Val Loss = {val_loss/len(val_loader):.4f}, '
                  f'Val Margin = {val_margin:.4f}')
    
    return best_model

def extract_indicator_species(model, feature_names, X, y, topn=10):
    """
    Extract indicator species based on feature importance in the first layer
    """
    model.eval()
    with torch.no_grad():
        # Get weights from first layer
        first_layer = model.layers[0]
        weights = first_layer.weight.data.abs()
        
        # Get mean importance of each input feature
        feature_importance = weights.mean(dim=0)
        
        # Get top features
        top_indices = torch.argsort(feature_importance, descending=True)[:topn]
        top_species = [feature_names[i] for i in top_indices]
        
        return top_species, feature_importance[top_indices]
