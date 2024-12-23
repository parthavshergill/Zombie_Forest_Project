import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
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

# Step 2: Reclassify composite_zf_class into binary classes
target = target.apply(lambda x: 'VCM' if 'VCM' in x else 'Non-VCM')

# Step 3: Create species co-occurrence features
species_presence = pd.get_dummies(species, prefix='species')
species_counts = pd.concat([grid_location, species_presence], axis=1).groupby('grid_location').sum()

# Step 4: Aggregate environmental features by grid_location
bio_features = pd.concat([grid_location, bio_features], axis=1).groupby('grid_location').mean()

# Step 5: One-hot encode cwhr_type and aggregate by grid_location
cwhr_one_hot = pd.get_dummies(cwhr_type, prefix='cwhr')
cwhr_features = pd.concat([grid_location, cwhr_one_hot], axis=1).groupby('grid_location').sum()

# Step 6: Map target variable to grid_location
target = pd.DataFrame({'grid_location': grid_location, 'composite_zf_class': target})
target = target.groupby('grid_location').first()

# Combine all features and target
features = species_counts.merge(bio_features, left_index=True, right_index=True)
features = features.merge(cwhr_features, left_index=True, right_index=True)
features = features.merge(target, left_index=True, right_index=True)

# Step 7: Encode the target variable
label_encoder = LabelEncoder()
features['composite_zf_class'] = label_encoder.fit_transform(features['composite_zf_class'])

# Separate features and target
X = features.drop(columns=['composite_zf_class'])
y = features['composite_zf_class']

# Split into train/test/validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Adjust class imbalance using RandomUnderSampler
# rus = RandomUnderSampler(random_state=42)
# X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Step 8: Train XGBoost with scale_pos_weight
scale_pos_weight = len(y_train_resampled) / sum(y_train_resampled) - 1
clf_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
clf_xgb.fit(X_train_resampled, y_train_resampled)

# Evaluate on validation set
y_val_pred = clf_xgb.predict(X_val_scaled)
print("\nValidation Set Performance:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

# Cross-validation for consistency
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf_xgb, X_train_scaled, y_train, cv=kfold, scoring='accuracy')
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Step 9: Evaluate final model on test set
y_test_pred = clf_xgb.predict(X_test_scaled)
print("\nTest Set Performance:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# # Step 10: Feature Importance
# feature_importances = clf_xgb.feature_importances_
# feature_names = X.columns
# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# # Display top 10 features
# top_features = importance_df.head(10)
# print("\nTop 10 Features:")
# print(top_features)

# # Visualization of feature importance
# plt.figure(figsize=(10, 6))
# plt.barh(top_features['Feature'], top_features['Importance'], color='blue')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.title('Top 10 Features for VCM Prediction')
# plt.gca().invert_yaxis()
# plt.show()

# Example: Extract feature importance from a trained XGBoost model
feature_importance = clf_xgb.get_booster().get_score(importance_type='weight')  # Default is 'weight'

# Types of importance
importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

# Retrieve and display all types of feature importance
for imp_type in importance_types:
    importance = clf_xgb.get_booster().get_score(importance_type=imp_type)
    print(f"{imp_type.capitalize()} Importance:")
    print(importance)

