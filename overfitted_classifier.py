import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

data = pd.read_csv('data_sources/inat-table-for-parthav.csv')  

# Identify unique values in 'conifer_vcm_class'
def unique_conifer_classes(data):
    return data['conifer_vcm_class'].unique()

conifer_classes = unique_conifer_classes(data)
print("Unique conifer_vcm_class values:", conifer_classes)

# Prepare data for classification
def prepare_data(data):
    df = data.dropna(subset=['conifer_vcm_class', 'habitat_suitability'])
    X = df[['habitat_suitability']].values
    y = pd.Categorical(df['conifer_vcm_class']).codes  # Encode labels numerically
    return X, y

X, y = prepare_data(data)

# Define classifier training with hyperparameter tuning
def train_classifier(X, y):
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    cv = KFold(n_splits=6, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

# Train and tune the model
best_model, best_params = train_classifier(X, y)
print("Best Parameters:", best_params)
print("Classification Report:")
print(classification_report(y, best_model.predict(X)))