import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(f"MLflow version: {mlflow.__version__}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

def configure_experiment():
    """Initialize or create MLflow experiment for CI"""
    name = "Basic_CI_Experiment"
    try:
        mlflow.set_experiment(name)
        print(f"✔️ Experiment '{name}' is now active.")
    except Exception as err:
        print(f"⚠️ Failed to set experiment: {err}")
        try:
            mlflow.create_experiment(name)
            mlflow.set_experiment(name)
            print(f"📌 Created and switched to experiment '{name}'")
        except Exception as err2:
            print(f"❌ Could not create experiment: {err2}")
            raise

def read_dataset():
    """Read dataset from preprocessing folder with validation"""
    try:
        base_path = "./dataset_preprocessing"
        files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
        for fname in files:
            path = os.path.join(base_path, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

        X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(base_path, "y_train.csv"))["target"].values
        y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))["target"].values

        print("📁 Dataset successfully loaded:")
        print(f"   Training data: {X_train.shape}, Labels: {len(y_train)}")
        print(f"   Test data: {X_test.shape}, Labels: {len(y_test)}")
        return X_train, X_test, y_train, y_test

    except Exception as err:
        print(f"❌ Failed to load dataset: {err}")
        print("🔍 Make sure the preprocessing step has been completed.")
        raise

def run_training():
    """Train Random Forest model and log results to MLflow"""
    print("🚦 Initiating basic CI model training...")
    print("-" * 50)

    configure_experiment()

    with mlflow.start_run(run_name="CI_Train_Run"):
        try:
            X_train, X_test, y_train, y_test = read_dataset()

            config = {
                "n_estimators": 50,
                "max_depth": 5,
                "random_state": 42,
                "n_jobs": 1
            }

            print("🧠 Training RandomForest model with config:")
            print(config)

            model = RandomForestClassifier(**config)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            acc = accuracy_score(y_test, predictions)

            mlflow.log_params(config)
            mlflow.log_metrics({
                "accuracy": acc,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X_train.shape[1]
            })

            cm = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["1", "0"], yticklabels=["1", "0"])
            plt.title("Confusion Matrix - Basic CI")
            plt.tight_layout()

            cm_filename = "conf_matrix_ci.png"
            plt.savefig(cm_filename, dpi=150)
            plt.close()

            mlflow.sklearn.log_model(model, "random_forest_model", registered_model_name="CI_RF_Model_Basic")
            mlflow.log_artifact(cm_filename)

            print("\n✅ Model training complete.")
            print(f"🔢 Accuracy: {acc:.4f}")
            print("📈 Classification Report:")
            print(classification_report(y_test, predictions, target_names=["1", "0"]))

            if os.path.exists(cm_filename):
                os.remove(cm_filename)

            return model, acc

        except Exception as err:
            print(f"❌ An error occurred during training: {err}")
            mlflow.log_param("error", str(err))
            raise

if __name__ == "__main__":
    print("🚀 Launching training process...")
    try:
        model, final_acc = run_training()
        print(f"\n🏁 Process finished. Final Accuracy: {final_acc:.4f}")
        sys.exit(0)
    except Exception as e:
        print(f"\n🔥 Training process failed: {e}")
        sys.exit(1)
