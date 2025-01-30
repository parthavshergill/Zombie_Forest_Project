from preprocess import load_and_preprocess_data, get_top_species
from deep_feature_learning import (VegetationDataset, OptimalFeatureLearning, 
                                 train_model, extract_indicator_species)
from torch.utils.data import DataLoader
import torch

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Create data loaders
    train_dataset = VegetationDataset(X_train, y_train)
    test_dataset = VegetationDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = OptimalFeatureLearning(input_dim)
    
    # Train model
    best_model_state = train_model(model, train_loader, test_loader, n_epochs=100)
    model.load_state_dict(best_model_state)
    
    # Extract indicator species
    top_species, importance_scores = extract_indicator_species(
        model, feature_names, X_test, y_test, topn=25
    )
    
    # Print results
    print("\nTop 25 Indicator Species:")
    for species, score in zip(top_species, importance_scores):
        print(f"{species.replace('species_', '')}: {score:.4f}")
    
    # Calculate final test accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            outputs, _ = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
