import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data_sources/complete-data-w-grid-locs.csv')

# Step 1: Extract relevant columns
species = data['species']
grid_location = data['grid_location']
bio_features = data[['bio1', 'bio12']]
cwhr_type = data['cwhr_type']
target = data['composite_zf_class']

# Step 2: Create species co-occurrence features
species_presence = pd.get_dummies(species, prefix='species')
species_counts = pd.concat([grid_location, species_presence], axis=1).groupby('grid_location').sum()

# Step 3: Aggregate environmental features by grid_location
bio_features = pd.concat([grid_location, bio_features], axis=1).groupby('grid_location').mean()

# Step 4: One-hot encode cwhr_type and aggregate by grid_location
cwhr_one_hot = pd.get_dummies(cwhr_type, prefix='cwhr')
cwhr_features = pd.concat([grid_location, cwhr_one_hot], axis=1).groupby('grid_location').sum()

# Step 5: Map target variable to grid_location
target = pd.DataFrame({'grid_location': grid_location, 'composite_zf_class': target})
target = target.groupby('grid_location').first()

# Combine all features and target
features = species_counts.merge(bio_features, left_index=True, right_index=True)
features = features.merge(cwhr_features, left_index=True, right_index=True)
features = features.merge(target, left_index=True, right_index=True)

# Convert the index (grid_location) into a column
features_reset = features.reset_index()

# Export the features matrix to CSV
features_reset.to_csv('feature_matrix.csv', index=False)

print("Feature matrix has been saved to feature_matrix.csv")

# Step 6: Encode the target variable
label_encoder = LabelEncoder()
features['composite_zf_class'] = label_encoder.fit_transform(features['composite_zf_class'])

# Split data
X = features.drop(columns=['composite_zf_class'])
y = features['composite_zf_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Step 7: Train models
# Logistic Regression
clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_lr.fit(X_train_balanced, y_train_balanced)
y_pred_lr = clf_lr.predict(X_test_scaled)

# Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf = clf_rf.predict(X_test_scaled)

# XGBoost
clf_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train_balanced, y_train_balanced)
y_pred_xgb = clf_xgb.predict(X_test_scaled)

# Step 8: Evaluate models
print("Logistic Regression:")
print(accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))

print("\nRandom Forest:")
print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

print("\nXGBoost:")
print(accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))

# Step 9: Visualization
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracies = [accuracy_score(y_test, y_pred_lr), 
              accuracy_score(y_test, y_pred_rf), 
              accuracy_score(y_test, y_pred_xgb)]

plt.bar(model_names, accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()
