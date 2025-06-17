import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime

# Ensure compatibility
print(f"MLflow version: {mlflow.__version__}")
print(f"Python version: {sys.version}")

def setup_mlflow():
    """Setup MLflow experiment for skilled level"""
    experiment_name = "CI_ML_Experiment_Skilled"
    
    try:
        mlflow.set_experiment(experiment_name)
        print(f"✅ Experiment '{experiment_name}' set successfully")
    except Exception as e:
        print(f"⚠️  Creating new experiment: {experiment_name}")
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(f"✅ Experiment created and set")
        except Exception as e2:
            print(f"❌ Failed to create experiment: {e2}")
            raise

def load_data():
    """Load preprocessed data with validation"""
    try:
        # Load data
        X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
        X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
        y_train = pd.read_csv('./dataset_preprocessing/y_train.csv')['target'].values
        y_test = pd.read_csv('./dataset_preprocessing/y_test.csv')['target'].values

        # Validation
        assert len(X_train) == len(y_train), "Training data length mismatch"
        assert len(X_test) == len(y_test), "Test data length mismatch"
        assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch"

        print(f"📊 Data loaded and validated:")
        print(f"   - Training: {X_train.shape}, Labels: {len(y_train)}")
        print(f"   - Testing: {X_test.shape}, Labels: {len(y_test)}")
        print(f"   - Features: {X_train.shape[1]}")
        print(f"   - Classes: {np.unique(y_train)}")

        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def create_comprehensive_artifacts(model, X_train, X_test, y_test, y_pred, run_id):
    """Create comprehensive artifacts for skilled level"""
    
    artifacts = {}
    
    # 1. Enhanced Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['1', '0'],
               yticklabels=['1', '0'])
    plt.title(f'Skilled CI - Confusion Matrix\nRun: {run_id}')
    
    # Add percentages
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'{cm[i,j]/total:.1%}',
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    cm_path = f'confusion_matrix_skilled_{run_id}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts['confusion_matrix'] = cm_path

    # 2. Feature Importance Plot
    feature_names = X_train.columns
    importances = model.feature_importances_

    plt.figure(figsize=(12, 8))
    sorted_idx = np.argsort(importances)[::-1]
    
    # Create bar plot
    bars = plt.bar(range(len(importances)), importances[sorted_idx], 
                   color='skyblue', alpha=0.7, edgecolor='navy')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], 
               rotation=45, ha='right')
    plt.title(f'Feature Importance - Skilled CI\nRun: {run_id}')
    plt.ylabel('Importance Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fi_path = f'feature_importance_skilled_{run_id}.png'
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts['feature_importance'] = fi_path

    # 3. Model Performance Summary Plot
    plt.figure(figsize=(10, 6))
    
    # Calculate metrics for visualization
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.title(f'Model Performance Metrics - Skilled CI\nRun: {run_id}')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    metrics_path = f'performance_metrics_skilled_{run_id}.png'
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts['performance_metrics'] = metrics_path

    return artifacts

def train_skilled_model():
    """Train model with comprehensive logging and artifacts"""
    
    print("🎯 Starting CI ML Training - Skilled Level")
    print("=" * 60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Generate unique run ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"CI_Skilled_Training_{run_id}"
    
    with mlflow.start_run(run_name=run_name):
        try:
            # Load data
            X_train, X_test, y_train, y_test = load_data()

            # Enhanced model configuration
            model_params = {
                'n_estimators': 100,
                'random_state': 42,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'n_jobs': 1  # Single job for CI stability
            }

            print(f"🤖 Training Enhanced Random Forest...")
            print(f"   Parameters: {model_params}")
            
            # Train model
            model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)

            # Cross-validation for better evaluation
            print("🔄 Performing 5-fold cross-validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Log parameters
            mlflow.log_params(model_params)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("run_id", run_id)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("cv_mean_accuracy", cv_mean)
            mlflow.log_metric("cv_std_accuracy", cv_std)

            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=f"PersonalityClassifier_CI_Skilled"
            )

            # Create and log comprehensive artifacts
            print("📊 Creating comprehensive artifacts...")
            artifacts = create_comprehensive_artifacts(
                model, X_train, X_test, y_test, y_pred, run_id
            )

            # Log all artifacts
            for artifact_name, artifact_path in artifacts.items():
                mlflow.log_artifact(artifact_path)
                print(f"   ✅ Logged: {artifact_name}")

            # Create and log detailed results JSON
            detailed_results = {
                "experiment_info": {
                    "run_id": run_id,
                    "run_name": run_name,
                    "timestamp": datetime.now().isoformat(),
                    "mlflow_version": mlflow.__version__
                },
                "model_config": model_params,
                "data_info": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "n_features": X_train.shape[1],
                    "feature_names": X_train.columns.tolist()
                },
                "performance_metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                },
                "cross_validation": {
                    "cv_scores": cv_scores.tolist(),
                    "cv_mean": cv_mean,
                    "cv_std": cv_std
                },
                "feature_importance": dict(zip(X_train.columns, model.feature_importances_))
            }

            results_path = f'skilled_results_{run_id}.json'
            with open(results_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            mlflow.log_artifact(results_path)
            print(f"   ✅ Logged: detailed_results")

            # Print comprehensive results
            print(f"\n=== SKILLED CI TRAINING RESULTS ===")
            print(f"🏆 Test Accuracy: {accuracy:.4f}")
            print(f"🎯 Precision: {precision:.4f}")
            print(f"📊 Recall: {recall:.4f}")
            print(f"🔄 F1-Score: {f1:.4f}")
            print(f"✅ CV Score: {cv_mean:.4f} ± {cv_std:.4f}")

            print(f"\n=== ARTIFACTS CREATED ===")
            for artifact_name, artifact_path in artifacts.items():
                print(f"📁 {artifact_name}: {artifact_path}")
            print(f"📄 Detailed results: {results_path}")

            print(f"\n=== CLASSIFICATION REPORT ===")
            print(classification_report(y_test, y_pred, target_names=['Introvert', 'Extrovert']))

            # Cleanup local files (artifacts are already logged to MLflow)
            for artifact_path in artifacts.values():
                if os.path.exists(artifact_path):
                    os.remove(artifact_path)
            
            if os.path.exists(results_path):
                os.remove(results_path)

            return model, accuracy, detailed_results

        except Exception as e:
            print(f"❌ Error during skilled training: {e}")
            mlflow.log_param("error", str(e))
            raise

if __name__ == "__main__":
    try:
        print("🚀 Starting Skilled CI-friendly ML training...")
        model, accuracy, results = train_skilled_model()
        print(f"\n🎉 Skilled training completed successfully!")
        print(f"🏆 Final accuracy: {accuracy:.4f}")
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        print(f"\n💥 Skilled training failed: {e}")
        # Exit with error code for CI
        sys.exit(1)