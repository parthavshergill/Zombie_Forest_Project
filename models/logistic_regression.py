import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Reload the dataset
file_path = 'data_sources/inat-data-matrix.csv'
data = pd.read_csv(file_path)

# Identify species columns and the target label
species_columns = [col for col in data.columns if col.startswith("species_")]
X = data[species_columns]
y = data['vcm_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights to address class imbalance
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

# Train a logistic regression classifier with class weights
logreg = LogisticRegression(class_weight=class_weight_dict, max_iter=1000, solver='liblinear', penalty='l1')
logreg.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC Score: {roc_auc}")