'''
Logistic Regression Classifier adjusting for class imbalance
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE  # Requires imbalanced-learn package

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

# Step 2: Add the target variable (composite_zf_class) to the pivot table
target = data.groupby('grid_location')['composite_zf_class'].first()
species_counts = species_counts.merge(target, left_index=True, right_index=True)

# Step 3: Encode the target variable
label_encoder = LabelEncoder()
species_counts['composite_zf_class'] = label_encoder.fit_transform(species_counts['composite_zf_class'])

# Step 4: Split the data into training and testing sets
X = species_counts.drop(columns=['composite_zf_class'])
y = species_counts['composite_zf_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Use SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Logistic Regression with class weights balanced
clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf_lr.fit(X_train_balanced, y_train_balanced)
y_pred_lr = clf_lr.predict(X_test)

# Evaluation for Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_)
print("Logistic Regression with Class Weights and SMOTE")
print(f"Accuracy: {accuracy_lr}")
print("Classification Report:\n", report_lr)

# Display the coefficients of the logistic regression model
species_coefficients = pd.DataFrame({
    'Species': X.columns,
    'Coefficient': clf_lr.coef_[0]  # Adjust for binary classification; modify if multi-class
})

# Sort by absolute value of coefficients to see most influential species
species_coefficients['Abs_Coefficient'] = species_coefficients['Coefficient'].abs()
species_coefficients = species_coefficients.sort_values(by='Abs_Coefficient', ascending=False)

# Display top coefficients
print("Logistic Regression Coefficients:")
print(species_coefficients[['Species', 'Coefficient']])

