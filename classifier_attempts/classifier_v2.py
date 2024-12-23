'''
Random Forest and Logistic Regression 
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'data_sources/complete-data-w-grid-locs.csv'  # Replace with actual file path
data = pd.read_csv(file_path)

# List of specified species to include as presence/absence indicators
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

# Step 1: Create presence/absence features for the selected species
# Create a pivot table indicating presence (1) or absence (0) of each selected species in each grid box
data['species_present'] = data['species'].apply(lambda x: 1 if x in selected_species else 0)
presence_absence = data[data['species_present'] == 1].pivot_table(index='grid_location', 
                                                                  columns='species', 
                                                                  values='species_present', 
                                                                  fill_value=0)

# Filter only the selected species as columns (others are dropped)
presence_absence = presence_absence.reindex(columns=selected_species, fill_value=0)

# Step 2: Add the target variable (composite_zf_class) to the pivot table
target = data.groupby('grid_location')['composite_zf_class'].first()
presence_absence = presence_absence.merge(target, left_index=True, right_index=True)

# Step 3: Encode the target variable
label_encoder = LabelEncoder()
presence_absence['composite_zf_class'] = label_encoder.fit_transform(presence_absence['composite_zf_class'])

# Step 4: Split the data into training and testing sets
X = presence_absence.drop(columns=['composite_zf_class'])
y = presence_absence['composite_zf_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# Evaluation for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_)
print("Random Forest Classifier")
print(f"Accuracy: {accuracy_rf}")
print("Classification Report:\n", report_rf)

# Logistic Regression
clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)

# Evaluation for Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_)
print("\nLogistic Regression")
print(f"Accuracy: {accuracy_lr}")
print("Classification Report:\n", report_lr)

# Display the coefficients of the logistic regression model
species_coefficients = pd.DataFrame({
    'Species': X.columns,
    'Coefficient': clf_lr.coef_[0]  # Assuming binary classification; adjust if multi-class
})

# Sort by absolute value of coefficients to see most influential species
species_coefficients['Abs_Coefficient'] = species_coefficients['Coefficient'].abs()
species_coefficients = species_coefficients.sort_values(by='Abs_Coefficient', ascending=False)

# Display top coefficients
print("Logistic Regression Coefficients:")
print(species_coefficients[['Species', 'Coefficient']])

