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

# 1. Load the dataset
file_path = 'data_sources/inat-data-matrix.csv'
data = pd.read_csv(file_path)

# 2. Prepare feature columns
species_columns = [col for col in data.columns if col.startswith("species_")]

#    Add the environmental columns bio1 and bio12
extra_features = ["bio1", "bio12"]

#    Convert veg_class into a binary variable
data["veg_class_binary"] = (data["veg_class"] == "Conifer-dominated").astype(int)

#    Combine all feature columns
feature_cols = species_columns # + extra_features # + ["veg_class_binary"]

X = data[feature_cols]
y = data["vcm_label"]

# 4. Split the data with stratification for class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use a custom scorer for ROC AUC
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Address class imbalance in XGBoost
# scale_pos_weight = (number of negative samples) / (number of positive samples)
count_neg = (y_train == 0).sum()
count_pos = (y_train == 1).sum()
scale_pos_weight = count_neg / float(count_pos)
print("scale_pos_weight =", scale_pos_weight)

# Define an XGBoost classifier
xgb_clf = XGBClassifier(
    objective="binary:logistic", 
    eval_metric="auc", 
    use_label_encoder=False,
    random_state=42
)

# Set up a parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    # Try both default (1) and computed scale_pos_weight to see if it helps
    'scale_pos_weight': [1, scale_pos_weight]
}

# Use GridSearchCV to tune hyperparameters
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring=roc_auc_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# Fit the model with grid search
grid_search.fit(X_train, y_train)

# Best hyperparameters and cross-validated ROC AUC
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validated ROC AUC: {grid_search.best_score_}")

# Train final XGBoost model with best hyperparameters
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train, y_train)

# Evaluate on the test set
y_pred = best_xgb.predict(X_test)
y_pred_prob = best_xgb.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance_vals = best_xgb.feature_importances_

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance_vals
}).sort_values('importance', ascending=False)

feature_importance.to_csv("xgb_species_feature_importance.csv", index=False)

print("Top 10 Most Important Features for Predicting VCM:")
print(feature_importance.head(10))


