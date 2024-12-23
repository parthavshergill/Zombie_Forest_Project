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

# Step 2: Reclassify composite_zf_class into binary classes
# Assuming VCM-related labels contain the term 'VCM' (adjust based on your actual labels)
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

# Step 8: Train XGBoost Model
clf_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train_balanced, y_train_balanced)
y_pred_xgb = clf_xgb.predict(X_test_scaled)

# Step 9: Evaluate the XGBoost Model
print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))

# Step 10: Extract Feature Importance
feature_importances = clf_xgb.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Filter for species-related features
species_importances = importance_df[importance_df['Feature'].str.startswith('species_')]

# Display top 10 species indicators
top_species = species_importances.head(10)
print("\nTop 10 Indicator Species for VCM Prediction:")
print(top_species)

# Step 11: Visualization of Top Features
plt.figure(figsize=(10, 6))
plt.barh(top_species['Feature'], top_species['Importance'], color='blue')
plt.xlabel('Importance')
plt.ylabel('Species')
plt.title('Top 10 Indicator Species for VCM Prediction')
plt.gca().invert_yaxis()
plt.show()
