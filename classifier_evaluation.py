import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from pathlib import Path
import argparse

def load_data(data_path):
    """Load and prepare data for classification."""
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Extract species columns and target
    species_columns = [col for col in data.columns if col.startswith('species_')]
    if 'vcm_label' not in data.columns:
        raise ValueError("vcm_label column not found in the data")
    
    print(f"Found {len(species_columns)} species columns")
    print(f"Dataset shape: {data.shape}")
    
    return data, species_columns

def load_indicator_species():
    """Load indicator species from different analysis outputs."""
    indicators = {}
    
    # 1. Load IndVal indicators
    try:
        indval_path = 'outputs/indval_analysis/top_indval_summary.csv'
        if os.path.exists(indval_path):
            indval_df = pd.read_csv(indval_path)
            indicators['indval'] = [f"species_{s}" for s in indval_df['Species'].tolist()]
            print(f"Loaded {len(indicators['indval'])} IndVal indicator species")
        else:
            print(f"Warning: IndVal file not found at {indval_path}")
            indicators['indval'] = []
    except Exception as e:
        print(f"Error loading IndVal indicators: {e}")
        indicators['indval'] = []
    
    # 2. Load Co-occurrence indicators
    try:
        cooccur_path = 'outputs/cooccurrence_analysis/top_indicators_summary.csv'
        if os.path.exists(cooccur_path):
            cooccur_df = pd.read_csv(cooccur_path)
            indicators['cooccurrence'] = [f"species_{s}" for s in cooccur_df['Species'].tolist()]
            print(f"Loaded {len(indicators['cooccurrence'])} Co-occurrence indicator species")
        else:
            print(f"Warning: Co-occurrence file not found at {cooccur_path}")
            indicators['cooccurrence'] = []
    except Exception as e:
        print(f"Error loading co-occurrence indicators: {e}")
        indicators['cooccurrence'] = []
    
    # 3. Load Frequency Analysis indicators (top 10)
    try:
        freq_path = 'outputs/frequency_analysis/significant_vcm_preferences.csv'
        if os.path.exists(freq_path):
            freq_df = pd.read_csv(freq_path)
            # Get top 10 VCM preference species
            top_freq = freq_df.sort_values('VCM_Preference_Strength', ascending=False).head(10)
            indicators['frequency'] = [f"species_{s}" for s in top_freq['Species'].tolist()]
            print(f"Loaded {len(indicators['frequency'])} Frequency indicator species")
        else:
            print(f"Warning: Frequency analysis file not found at {freq_path}")
            indicators['frequency'] = []
    except Exception as e:
        print(f"Error loading frequency indicators: {e}")
        indicators['frequency'] = []
    
    # 4. Create a merged set of all unique indicators
    all_indicators = list(set(indicators['indval'] + indicators['cooccurrence'] + indicators['frequency']))
    indicators['combined'] = all_indicators
    print(f"Combined unique indicators: {len(indicators['combined'])}")
    print(f"Unique indicators: {indicators['combined']}")
    
    return indicators

def train_evaluate_model(X_train, X_test, y_train, y_test, model_name="Model"):
    """Train and evaluate an XGBoost model on the given data."""
    print(f"\nTraining {model_name} with {X_train.shape[1]} features...")
    
    # Initialize model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), 
                               cv=cv, scoring='roc_auc')
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_roc_auc_mean': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std(),
        'feature_count': X_train.shape[1]
    }
    
    print(f"{model_name} metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Cross-Val ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return metrics, model, y_pred, y_prob, feature_importance

def plot_roc_curves(y_test, model_probs, model_names, output_dir):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for i, (name, probs) in enumerate(zip(model_names, model_probs)):
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(y_test, model_preds, model_names, output_dir):
    """Plot confusion matrices for all models."""
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models*4, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (name, preds, ax) in enumerate(zip(model_names, model_preds, axes)):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'{name}')
        ax.set_xticklabels(['Non-VCM', 'VCM'])
        ax.set_yticklabels(['Non-VCM', 'VCM'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(metrics_list, output_dir):
    """Plot comparison of key metrics across models."""
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.set_index('model_name')
    
    # Select metrics to plot
    plot_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    plot_df = metrics_df[plot_metrics]
    
    plt.figure(figsize=(12, 8))
    ax = plot_df.plot(kind='bar', rot=0, figsize=(12, 8))
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.title('Model Performance Comparison', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.05)  # Set y-axis limit to accommodate labels
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Metrics')
    
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(feature_importance_df, model_name, output_dir, top_n=20):
    """Plot feature importance for a model."""
    # Take top N features
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x='Importance', y='Feature', data=top_features)
    
    plt.title(f'Top {top_n} Features - {model_name}', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # Add value labels
    for i, v in enumerate(top_features['Importance']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    plt.savefig(os.path.join(output_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def run_evaluation(data_path, output_dir):
    """Run the full evaluation pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data, all_species_columns = load_data(data_path)
    
    # Load indicator species
    indicators = load_indicator_species()
    
    # Check if we have any indicators
    if all(len(indicators[key]) == 0 for key in ['indval', 'cooccurrence', 'frequency']):
        print("Warning: No indicator species found in any of the expected files.")
        return
    
    # Prepare features and target
    X = data[all_species_columns]
    y = data['vcm_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"VCM label distribution in training: {y_train.value_counts().to_dict()}")
    
    # Train and evaluate baseline model (all species)
    baseline_metrics, baseline_model, baseline_preds, baseline_probs, baseline_importance = train_evaluate_model(
        X_train, X_test, y_train, y_test, "Baseline (All Species)"
    )
    
    # Save baseline feature importance
    plot_feature_importance(baseline_importance, "Baseline Model", output_dir)
    baseline_importance.to_csv(os.path.join(output_dir, 'baseline_feature_importance.csv'), index=False)
    
    # Initialize lists to store results
    all_metrics = [baseline_metrics]
    all_predictions = [baseline_preds]
    all_probabilities = [baseline_probs]
    all_model_names = ["Baseline"]
    
    # Train and evaluate indicator-based models
    for indicator_type, indicator_columns in indicators.items():
        if not indicator_columns:
            continue
            
        # Filter to get only existing columns
        valid_indicators = [col for col in indicator_columns if col in X.columns]
        
        if not valid_indicators:
            print(f"Warning: No valid indicator columns found for {indicator_type}")
            continue
            
        model_name = f"{indicator_type.capitalize()} Indicators"
        
        # Train model on indicator species
        X_train_indicators = X_train[valid_indicators]
        X_test_indicators = X_test[valid_indicators]
        
        metrics, model, y_pred, y_prob, feature_imp = train_evaluate_model(
            X_train_indicators, X_test_indicators, y_train, y_test, model_name
        )
        
        all_metrics.append(metrics)
        all_predictions.append(y_pred)
        all_probabilities.append(y_prob)
        all_model_names.append(model_name)
        
        # Save feature importance
        plot_feature_importance(feature_imp, model_name, output_dir)
        feature_imp.to_csv(os.path.join(output_dir, f'{indicator_type}_feature_importance.csv'), index=False)
    
    # Create visualizations
    plot_roc_curves(y_test, all_probabilities, all_model_names, output_dir)
    plot_confusion_matrices(y_test, all_predictions, all_model_names, output_dir)
    plot_metrics_comparison(all_metrics, output_dir)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'model_performance_metrics.csv'), index=False)
    
    # Generate a Venn diagram of indicator species overlap
    try:
        from matplotlib_venn import venn3
        
        if all(len(indicators[key]) > 0 for key in ['indval', 'cooccurrence', 'frequency']):
            plt.figure(figsize=(10, 10))
            
            # Strip 'species_' prefix for cleaner labels
            indval_set = set([s.replace('species_', '') for s in indicators['indval']])
            cooccur_set = set([s.replace('species_', '') for s in indicators['cooccurrence']])
            freq_set = set([s.replace('species_', '') for s in indicators['frequency']])
            
            v = venn3([indval_set, cooccur_set, freq_set], 
                    ('IndVal', 'Co-occurrence', 'Frequency'))
            
            plt.title('Overlap of Indicator Species Across Methods', fontsize=16)
            plt.savefig(os.path.join(output_dir, 'indicator_species_overlap.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except ImportError:
        print("matplotlib_venn not installed. Skipping Venn diagram creation.")
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate indicator species effectiveness for VCM prediction')
    parser.add_argument('--input', type=str, default='inat-data-matrix-latlong.csv',
                        help='Path to input data matrix CSV file')
    parser.add_argument('--output-dir', type=str, default='outputs/classifier_analysis',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    
    print("\n===== Indicator Species Effectiveness Evaluation =====")
    run_evaluation(args.input, args.output_dir)

if __name__ == "__main__":
    main() 