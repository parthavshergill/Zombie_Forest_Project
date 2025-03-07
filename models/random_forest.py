import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# Define our top indicator species
top_indicator_species = [
    'species_Toxicodendron diversilobum',
    'species_Silene laciniata',
    'species_Chamaebatia foliolosa',
    'species_Arctostaphylos viscida',
    'species_Adelinia grande',
    'species_Heteromeles arbutifolia',
    'species_Aesculus californica',
    'species_Elgaria multicarinata',
    'species_Calochortus albus',
    'species_Woodwardia fimbriata',
    'species_Lonicera hispidula',
    'species_Lysimachia latifolia',
    'species_Calochortus venustus',
    'species_Acer macrophyllum',
    'species_Chlorogalum pomeridianum',
    'species_Diplacus grandiflorus',
    'species_Ceanothus integerrimus',
    'species_Dicentra formosa',
    'species_Crotalus oreganus',
    'species_Iris hartwegii'
]

# Load data
print("Loading data...")
data = pd.read_csv("inat-data-matrix-gdf.csv")

# Prepare features and target
species_columns = [col for col in data.columns if col.startswith('species_')]
X_all = data[species_columns]
y = data['vcm_label']

# Calculate class weights
class_counts = y.value_counts()
total_samples = len(y)
class_weights = {
    0: total_samples / (2 * class_counts[0]),
    1: total_samples / (2 * class_counts[1])
}
print("\nClass weights:", class_weights)

# Prepare top indicators dataset
X_top = data[top_indicator_species]

# Split data
X_all_train, X_all_test, X_top_train, X_top_test, y_train, y_test = train_test_split(
    X_all, X_top, y, test_size=0.2, random_state=42, stratify=y
)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample', class_weights]
}

# Define scoring metric
scoring = {
    'AUC': 'roc_auc',
    'F1': 'f1',
    'Precision': 'precision',
    'Recall': 'recall'
}

def train_and_evaluate_model(X_train, X_test, name=""):
    """Helper function to train and evaluate a model with grid search"""
    print(f"\nTraining {name} model...")
    
    # Initialize base model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Print results
    print(f"\n{name} Model Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.3f}")
    print(f"Test set ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_model, grid_search.best_score_

# Train and evaluate both models
model_all, cv_score_all = train_and_evaluate_model(X_all_train, X_all_test, "All Species")
model_top, cv_score_top = train_and_evaluate_model(X_top_train, X_top_test, "Top Indicators")

# Feature importance for top indicators model
importance_df = pd.DataFrame({
    'species': [s.replace('species_', '') for s in top_indicator_species],
    'importance': model_top.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='importance', y='species')
plt.title('Feature Importance of Top Indicator Species')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('top_indicators_importance.png')

# Save results
importance_df.to_csv('top_indicators_importance.csv', index=False)

print("\nResults saved to:")
print("- top_indicators_importance.png")
print("- top_indicators_importance.csv")
