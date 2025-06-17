import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure compatibility
print(f"MLflow version: {mlflow.__version__}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

def setup_mlflow():
    """Setup MLflow experiment"""
    experiment_name = "CI_ML_Experiment_Basic"
    
    try:
        # Try to set experiment
        mlflow.set_experiment(experiment_name)
        print(f"✅ Experiment '{experiment_name}' set successfully")
    except Exception as e:
        print(f"⚠️  Error setting experiment: {e}")
        # Create experiment if it doesn't exist
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(f"✅ Experiment '{experiment_name}' created and set")
        except Exception as e2:
            print(f"❌ Failed to create experiment: {e2}")
            raise

def load_data():
    """Load preprocessed data with error handling"""
    try:
        # Check if files exist
        files_to_check = [
            './dataset_preprocessing/X_train.csv',
            './dataset_preprocessing/X_test.csv', 
            './dataset_preprocessing/y_train.csv',
            './dataset_preprocessing/y_test.csv'
        ]
        
        for file in files_to_check:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
        
        # Load data
        X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
        X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
        y_train = pd.read_csv('./dataset_preprocessing/y_train.csv')['target'].values
        y_test = pd.read_csv('./dataset_preprocessing/y_test.csv')['target'].values

        print(f"📊 Data loaded successfully:")
        print(f"   - Training set: {X_train.shape}")
        print(f"   - Test set: {X_test.shape}")
        print(f"   - Training labels: {len(y_train)}")
        print(f"   - Test labels: {len(y_test)}")
        print(f"   - Features: {list(X_train.columns)}")

        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Make sure you have copied dataset files from kriteria 1")
        raise

def train_model():
    """Train model with CI-friendly configuration"""
    
    print("🎯 Starting CI ML Training - Basic Level")
    print("=" * 50)
    
    # Setup MLflow
    setup_mlflow()
    
    with mlflow.start_run(run_name="CI_Basic_Training"):
        try:
            # Load data
            X_train, X_test, y_train, y_test = load_data()

            # Model configuration - simple for CI
            model_params = {
                'n_estimators': 50,  # Reduced for faster CI
                'random_state': 42,
                'max_depth': 5,      # Simplified for CI
                'n_jobs': 1          # Single job for CI stability
            }

            print("🤖 Training Random Forest model...")
            print(f"   Parameters: {model_params}")
            
            # Train model
            model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Log parameters and metrics
            mlflow.log_params(model_params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("n_features", X_train.shape[1])

            # Create simple confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['1', '0'],
                       yticklabels=['1', '0'])
            plt.title('CI Basic - Confusion Matrix')
            plt.tight_layout()
            
            # Save plot
            cm_path = 'confusion_matrix_ci_basic.png'
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to save memory in CI
            
            # Log model and artifacts
            mlflow.sklearn.log_model(model, "model", registered_model_name="PersonalityClassifier_CI_Basic")
            mlflow.log_artifact(cm_path)

            # Print results
            print(f"\n=== CI TRAINING RESULTS ===")
            print(f"✅ Model trained successfully!")
            print(f"🎯 Accuracy: {accuracy:.4f}")
            print(f"📊 Confusion Matrix saved: {cm_path}")
            
            print(f"\n=== CLASSIFICATION REPORT ===")
            print(classification_report(y_test, y_pred, target_names=['1', '0']))

            # Cleanup
            if os.path.exists(cm_path):
                os.remove(cm_path)

            return model, accuracy

        except Exception as e:
            print(f"❌ Error during training: {e}")
            mlflow.log_param("error", str(e))
            raise

if __name__ == "__main__":
    try:
        print("🚀 Starting CI-friendly ML training...")
        model, accuracy = train_model()
        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Final accuracy: {accuracy:.4f}")
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        # Exit with error code for CI
        sys.exit(1)