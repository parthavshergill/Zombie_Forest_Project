import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'data_sources/complete-data-w-grid-locs.csv'  # Replace with actual file path
data = pd.read_csv(file_path)

# Step 1: Create a pivot table to count species per grid box
pivot_table = data.pivot_table(index='grid_location', columns='species', aggfunc='size', fill_value=0)

# Step 2: Add the target variable (conifer_class) to the pivot table
# Group by grid_location to assign a unique conifer_class per grid_location
target = data.groupby('grid_location')['conifer_class'].first()
pivot_table = pivot_table.merge(target, left_index=True, right_index=True)

# Step 3: Encode the target variable
label_encoder = LabelEncoder()
pivot_table['conifer_class'] = label_encoder.fit_transform(pivot_table['conifer_class'])

# Step 4: Split the data into training and testing sets
X = pivot_table.drop(columns=['conifer_class'])
y = pivot_table['conifer_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the random forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)