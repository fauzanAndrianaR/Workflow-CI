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
from pathlib import Path

# Tampilkan versi package utama
print(f"🐍 Python version: {sys.version}")
print(f"📈 MLflow version: {mlflow.__version__}")
print(f"🧪 Scikit-learn version: {__import__('sklearn').__version__}")

# Konfigurasi backend matplotlib untuk lingkungan CI
plt.switch_backend('Agg')

# Aktifkan autolog untuk mencatat otomatis model dan metrik
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

def configure_mlflow():
    """Set URI tracking dan nama eksperimen"""
    uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
    mlflow.set_tracking_uri(uri)

    exp_name = "CI_HeartFailure_Classification"
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        mlflow.create_experiment(exp_name)
        print(f"🆕 Experiment dibuat: {exp_name}")
    else:
        print(f"📁 Menggunakan experiment lama: {exp_name}")
    mlflow.set_experiment(exp_name)
    return exp_name

def import_data():
    """Ambil data preprocessed atau buat dummy jika gagal"""
    try:
        path = Path('./dataset_preprocessing')
        if not path.exists():
            raise FileNotFoundError("Folder dataset tidak ditemukan")

        X_train = pd.read_csv(path / 'X_train.csv')
        X_test = pd.read_csv(path / 'X_test.csv')
        y_train = pd.read_csv(path / 'y_train.csv')['target'].values
        y_test = pd.read_csv(path / 'y_test.csv')['target'].values

        print("✅ Dataset berhasil dimuat.")
        return X_train, X_test, y_train, y_test

    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        return generate_dummy_data()

def generate_dummy_data():
    """Membuat data palsu untuk pengujian pipeline"""
    np.random.seed(0)
    n_samples = 1000
    n_features = 10

    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'feat_{i}' for i in range(n_features)])
    y = np.random.randint(0, 2, size=n_samples)

    return X[:800], X[800:], y[:800], y[800:]

def save_visuals(y_true, y_pred, importances, feat_names, run_id):
    """Simpan visualisasi confusion matrix dan feature importance"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
    plt.title('Confusion Matrix - Heart Failure')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_file = f'cm_{run_id[:8]}.png'
    plt.savefig(cm_file)
    plt.close()

    plt.figure(figsize=(10, 5))
    idx_sorted = np.argsort(importances)[::-1][:10]
    plt.bar(range(len(idx_sorted)), importances[idx_sorted])
    plt.xticks(range(len(idx_sorted)), [feat_names[i] for i in idx_sorted], rotation=45)
    plt.title('Top 10 Feature Importances')
    fi_file = f'fi_{run_id[:8]}.png'
    plt.savefig(fi_file)
    plt.close()

    return cm_file, fi_file

def train_model():
    """Training model klasifikasi heart failure dan logging MLflow"""
    experiment_name = configure_mlflow()
    with mlflow.start_run(run_name=f"RandomForest_Heart_{pd.Timestamp.now():%Y%m%d_%H%M%S}") as run:
        run_id = run.info.run_id
        X_train, X_test, y_train, y_test = import_data()

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        params = {
            'n_estimators': 120,
            'max_depth': 8,
            'random_state': 42,
            'max_features': 'sqrt'
        }
        for p, v in params.items():
            mlflow.log_param(p, v)

        print("🚀 Training model Random Forest...")
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        print(f"\n📊 Classification Report:\n{classification_report(y_test, preds, target_names=['No Failure', 'Failure'])}")

        try:
            cm_path, fi_path = save_visuals(y_test, preds, clf.feature_importances_, X_train.columns, run_id)
            mlflow.log_artifact(cm_path, "plots")
            mlflow.log_artifact(fi_path, "plots")
            os.remove(cm_path)
            os.remove(fi_path)
        except Exception as vis_err:
            print(f"⚠️ Gagal membuat visualisasi: {vis_err}")

        mlflow.set_tag("dataset", "heart_failure")
        mlflow.set_tag("type", "binary_classification")
        mlflow.set_tag("framework", "sklearn")

        return clf, acc, run_id

def main():
    print("💓 Heart Failure Classification Pipeline")
    print("=" * 60)
    try:
        model, acc, run_id = train_model()
        print(f"\n🎉 SUCCESS! Accuracy: {acc:.4f} | Run ID: {run_id}")
        print("📍 Jalankan `mlflow ui` untuk melihat hasil.")
        return 0
    except Exception as err:
        print(f"💥 ERROR: {err}")
        try:
            mlflow.log_param("pipeline_error", str(err))
        except:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())
