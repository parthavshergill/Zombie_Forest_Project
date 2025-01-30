import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath='inatdatamatrix.csv'):
    """
    Load and preprocess the vegetation classification data
    """
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Separate features (species) and target (vcm_label)
    species_cols = [col for col in df.columns if col.startswith('species_')]
    X = df[species_cols]
    y = df['vcm_label']
    
    # Convert matrix to float type
    X = X.astype(float)
    
    # Handle missing values if any
    X = X.fillna(0)
    
    # Apply log transformation to species abundance data
    X = np.log1p(X)  # log1p(x) = log(1 + x)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns

def get_top_species(X, n_top=50):
    """
    Get top n species by total abundance
    """
    species_sums = X.sum()
    top_species = species_sums.nlargest(n_top).index
    return top_species

if __name__ == "__main__":
    # Test preprocessing
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    
    # Get top 50 species
    top_species = get_top_species(X_train)
    print("\nTop 10 species by abundance:")
    print(top_species[:10])
