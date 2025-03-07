import pandas as pd
import numpy as np

# XGBoost
from xgboost import XGBClassifier

# Scikit-learn utilities
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score, classification_report

# For plotting
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# SHAP for model explainability
import shap

# Define top indicator species
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

# 1. Load the dataset
print("Loading data...")
file_path = 'inat-data-matrix-gdf.csv'
data = pd.read_csv(file_path)

# Add this debugging code after loading the data
print("\nChecking available species columns...")
species_cols = [col for col in data.columns if col.startswith('species_')]
print("\nLooking for 'Lonicera' in columns:")
lonicera_cols = [col for col in species_cols if 'Lonicera' in col]
print(lonicera_cols)

# Also check if all our top indicators are present
print("\nMissing species from our top indicators:")
missing = [sp for sp in top_indicator_species if sp not in species_cols]
print(missing)

# 2. Prepare feature columns
species_columns = [col for col in data.columns if col.startswith("species_")]

# Create two feature sets
X_all = data[species_columns]
X_top = data[top_indicator_species]
y = data["vcm_label"]

# 4. Split the data with stratification for class balance
X_all_train, X_all_test, X_top_train, X_top_test, y_train, y_test = train_test_split(
    X_all, X_top, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Calculate class weights
count_neg = (y_train == 0).sum()
count_pos = (y_train == 1).sum()
scale_pos_weight = count_neg / float(count_pos)
print(f"\nClass balance (scale_pos_weight): {scale_pos_weight}")

# Set up parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, scale_pos_weight]
}

def train_and_evaluate_model(X_train, X_test, name=""):
    """Helper function to train and evaluate a model"""
    print(f"\nTraining {name} model...")
    
    # Initialize model
    xgb_clf = XGBClassifier(
        objective="binary:logistic", 
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42
    )
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring=roc_auc_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\n{name} Model Results:")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best Cross-Validated ROC AUC: {grid_search.best_score_:.3f}")
    print(f"Test Set ROC AUC: {roc_auc_score(y_test, y_pred_prob):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_model, grid_search.best_score_

# Train and evaluate both models
model_all, cv_score_all = train_and_evaluate_model(X_all_train, X_all_test, "All Species")
model_top, cv_score_top = train_and_evaluate_model(X_top_train, X_top_test, "Top Indicators")

# Feature importance for top indicators model
feature_importance = pd.DataFrame({
    'feature': top_indicator_species,
    'importance': model_top.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Indicator Species Importance:")
print(feature_importance)

# Save feature importance
feature_importance.to_csv("xgb_top_indicators_importance.csv", index=False)

# Visualize feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance['importance'])
plt.xticks(range(len(feature_importance)), 
           [s.replace('species_', '') for s in feature_importance['feature']], 
           rotation=45, ha='right')
plt.title('XGBoost Feature Importance - Top Indicators')
plt.tight_layout()
plt.savefig('xgb_top_indicators_importance.png')

# SHAP analysis for top indicators model
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model_top)
shap_values = explainer.shap_values(X_top_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_top_test, 
                 feature_names=[s.replace('species_', '') for s in top_indicator_species],
                 show=False)
plt.tight_layout()
plt.savefig('xgb_top_indicators_shap.png')

print("\nResults saved to:")
print("- xgb_top_indicators_importance.csv")
print("- xgb_top_indicators_importance.png")
print("- xgb_top_indicators_shap.png")


