'''
Using Max Class Loss as the loss function to minimize
'''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the dataset
file_path = 'data_sources/complete-data-w-grid-locs.csv'  # Replace with actual file path
data = pd.read_csv(file_path)

# List of specified species to include as count features
selected_species = {
    'Elgaria multicarinata', 'Sciurus griseus', 'Lepus californicus', 'Quercus kelloggii', 
    'Cytisus scoparius', 'Urocyon cinereoargenteus', 'Calochortus leichtlinii', 
    'Woodwardia fimbriata', 'Lilium parvum', 'Pterospora andromedea', 'Quercus chrysolepis', 
    'Veratrum californicum', 'Chamaenerion angustifolium', 'Pentagramma triangularis', 
    'Rubus armeniacus', 'Odocoileus hemionus', 'Toxicodendron diversilobum', 
    'Aesculus californica', 'Calochortus albus', 'Sarcodes sanguinea', 'Collinsia tinctoria', 
    'Lonicera hispidula', 'Penstemon newberryi', 'Taricha sierrae', 'Arbutus menziesii', 
    'Meleagris gallopavo', 'Melanerpes formicivorus', 'Calochortus monophyllus', 
    'Platanthera dilatata', 'Sedum spathulifolium', 'Heteromeles arbutifolia', 
    'Lilium humboldtii', 'Sequoiadendron giganteum', 'Chamaebatia foliolosa', 
    'Silene laciniata', 'Arctostaphylos viscida', 'Adelinia grande', 'Lathyrus sulphureus'
}

# Step 1: Create count features for the selected species
species_counts = data[data['species'].isin(selected_species)].pivot_table(
    index='grid_location', 
    columns='species', 
    aggfunc='size',  # Count occurrences of each species
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

# Step 5: Split the data into training and testing sets
X = features.drop(columns=['composite_zf_class'])
y = features['composite_zf_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Use SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Step 7: Calculate class weights based on maximum class loss approximation
class_counts = np.bincount(y_train_balanced)
class_weights = {i: max(class_counts) / count for i, count in enumerate(class_counts)}

# Logistic Regression with custom class weights
clf_lr = LogisticRegression(max_iter=1000, class_weight=class_weights, random_state=42)
clf_lr.fit(X_train_balanced, y_train_balanced)
y_pred_lr = clf_lr.predict(X_test)

# Random Forest with custom class weights
clf_rf = RandomForestClassifier(random_state=42, class_weight=class_weights)
clf_rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf = clf_rf.predict(X_test)

# XGBoost with scale_pos_weight
scale_pos_weights = [class_weights[i] for i in range(len(class_weights))]
clf_xgb = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weights)
clf_xgb.fit(X_train_balanced, y_train_balanced)
y_pred_xgb = clf_xgb.predict(X_test)

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