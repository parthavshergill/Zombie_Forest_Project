import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score, classification_report
import numpy as np

# Load the dataset
file_path = 'data_sources/inat-data-matrix.csv'
data = pd.read_csv(file_path)

# Identify species columns and the target label
species_columns = [col for col in data.columns if col.startswith("species_")]
X = data[species_columns]
y = data['vcm_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the Random Forest Classifier
rf = RandomForestClassifier(class_weight="balanced", random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define custom scorer for ROC AUC
roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=roc_auc_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# Fit the model with GridSearchCV
grid_search.fit(X_train, y_train)

# Display the best hyperparameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Display the best cross-validated ROC AUC score
print(f"Best Cross-Validated ROC AUC: {grid_search.best_score_}")

# Train the final model using the best hyperparameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]

# Display the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Extract feature importance
feature_importance = pd.DataFrame({
    'species': species_columns,
    'importance': best_rf.feature_importances_
}).sort_values(by='importance', ascending=False)

# Save feature importance to a CSV file
feature_importance.to_csv("rf_species_feature_importance_tuned.csv", index=False)

# Display the top 10 most important species
top_important_species = feature_importance.head(10)
print("Top 10 Most Important Species for Predicting VCM:")
print(top_important_species)
