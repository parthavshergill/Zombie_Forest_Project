import shap
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

# Step 9: SHAP Values Calculation
# Use SHAP TreeExplainer
explainer = shap.TreeExplainer(clf_xgb)
shap_values = explainer.shap_values(X_train_resampled)

# Ensure X_train_resampled is a DataFrame with the correct feature names
X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)

# Global Feature Importance with SHAP
shap.summary_plot(shap_values, X_train_resampled_df, feature_names=X.columns)
plt.title("SHAP Global Feature Importance")
plt.show()

# Display Top Features in a DataFrame
shap_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Mean SHAP Value': abs(shap_values).mean(axis=0)
}).sort_values(by='Mean SHAP Value', ascending=False)

print("\nTop Features by SHAP Values:")
print(shap_feature_importance.head(10))

# Visualize Top Features
top_features = shap_feature_importance.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Mean SHAP Value'], color='blue')
plt.xlabel('Mean SHAP Value')
plt.ylabel('Feature')
plt.title('Top 10 Features for VCM Prediction (SHAP)')
plt.gca().invert_yaxis()
plt.show()
