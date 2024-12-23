import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the dataset
data_file_path = 'data_sources/complete-data-sample.csv'  # Replace with your actual path
species_stats_file_path = 'outputs/species_stats.csv'  # Replace with your actual path
data = pd.read_csv(data_file_path)

# Load the species list from species_stats.csv
species_stats = pd.read_csv(species_stats_file_path)
selected_species = set(species_stats['species'].unique())

# Step 1: Create count features for the selected species
species_counts = data[data['species'].isin(selected_species)].pivot_table(
    index='grid_location',
    columns='species',
    aggfunc='size',
    fill_value=0
)

# Filter to only include the selected species as columns
species_counts = species_counts.reindex(columns=selected_species, fill_value=0)

# Step 2: Add bio1 and bio12 to the features
bio_features = data.groupby('grid_location')[['bio1', 'bio12']].mean()
features = species_counts.merge(bio_features, left_index=True, right_index=True)

# Step 3: Add the target variable (composite_zf_class) to the features
target = data.groupby('grid_location')['composite_zf_class'].first()
features = features.merge(target, left_index=True, right_index=True)

# Step 4: Encode the target variable
label_encoder = LabelEncoder()
features['composite_zf_class'] = label_encoder.fit_transform(features['composite_zf_class'])

features_reset = features.reset_index()

# Export the features matrix to CSV
features_reset.to_csv('feature_matrix.csv', index=False)

print("Features matrix has been saved to feature_matrix.csv")

# Step 5: Split the data into training and testing sets
X = features.drop(columns=['composite_zf_class'])
y = features['composite_zf_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Data prepared")

# Step 6: Use SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Step 7: Feature selection using RFE
clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rfe = RFE(estimator=clf_lr, n_features_to_select=20)  # Select top 20 features
rfe.fit(X_train_balanced, y_train_balanced)

# Get the selected features
selected_features = X_train_balanced.columns[rfe.support_]
X_train_selected = X_train_balanced[selected_features]
X_test_selected = X_test[selected_features]

# Train classifiers on selected features
# Logistic Regression
clf_lr.fit(X_train_selected, y_train_balanced)
y_pred_lr = clf_lr.predict(X_test_selected)

# Random Forest
clf_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf_rf.fit(X_train_selected, y_train_balanced)
y_pred_rf = clf_rf.predict(X_test_selected)

# XGBoost
scale_pos_weights = [max(np.bincount(y_train_balanced)) / count for count in np.bincount(y_train_balanced)]
clf_xgb = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weights)
clf_xgb.fit(X_train_selected, y_train_balanced)
y_pred_xgb = clf_xgb.predict(X_test_selected)

# Evaluation
print("Logistic Regression")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print("Classification Report:\n", classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))

print("\nRandom Forest")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

print("\nXGBoost")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))
