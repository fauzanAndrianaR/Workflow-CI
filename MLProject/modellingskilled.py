import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime

print(f"Using MLflow version: {mlflow.__version__}")
print(f"Python interpreter: {sys.version}")

def initialize_experiment():
    """Menyiapkan eksperimen MLflow"""
    experiment_title = "Skilled_Level_Experiment_CI"

    try:
        mlflow.set_experiment(experiment_title)
        print(f"✅ Experiment '{experiment_title}' sudah aktif")
    except Exception:
        print(f"⚠️  Gagal set, mencoba membuat baru: {experiment_title}")
        try:
            mlflow.create_experiment(experiment_title)
            mlflow.set_experiment(experiment_title)
            print("✅ Eksperimen berhasil dibuat dan diatur")
        except Exception as err:
            print(f"❌ Gagal membuat experiment baru: {err}")
            raise

def read_dataset():
    """Membaca dataset yang telah diproses sebelumnya"""
    try:
        X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
        X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
        y_train = pd.read_csv('./dataset_preprocessing/y_train.csv')['target'].values
        y_test = pd.read_csv('./dataset_preprocessing/y_test.csv')['target'].values

        assert len(X_train) == len(y_train), "Jumlah data training dan label tidak sesuai"
        assert len(X_test) == len(y_test), "Jumlah data testing dan label tidak sesuai"
        assert X_train.shape[1] == X_test.shape[1], "Jumlah fitur tidak konsisten"

        print("📥 Data berhasil dimuat:")
        print(f"   - Train: {X_train.shape}, Label: {len(y_train)}")
        print(f"   - Test : {X_test.shape}, Label: {len(y_test)}")
        print(f"   - Fitur : {X_train.shape[1]}")
        print(f"   - Kelas : {np.unique(y_train)}")

        return X_train, X_test, y_train, y_test

    except Exception as err:
        print(f"❌ Gagal membaca data: {err}")
        raise

def generate_artifacts(clf_model, X_train, X_test, y_test, pred, run_id):
    """Membuat dan menyimpan artefak hasil pelatihan"""
    artifacts = {}

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['1', '0'], yticklabels=['1', '0'])
    plt.title(f'Confusion Matrix\nRun ID: {run_id}')

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, f'{cm[i, j]/total:.1%}', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    cm_file = f'conf_matrix_{run_id}.png'
    plt.savefig(cm_file)
    plt.close()
    artifacts['conf_matrix'] = cm_file

    # Feature importance
    features = X_train.columns
    importance = clf_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(importance)), importance[sorted_idx], color='lightblue', edgecolor='blue')

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{bar.get_height():.3f}', ha='center', fontsize=8)

    plt.xticks(range(len(features)), [features[i] for i in sorted_idx], rotation=45, ha='right')
    plt.title(f'Feature Importance\nRun ID: {run_id}')
    plt.tight_layout()

    fi_file = f'feature_importance_{run_id}.png'
    plt.savefig(fi_file)
    plt.close()
    artifacts['feature_importance'] = fi_file

    # Performance metrics
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average='weighted')
    rec = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')

    plt.figure(figsize=(8, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [acc, prec, rec, f1]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = plt.bar(metrics, scores, color=colors, edgecolor='black')

    for bar, val in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center')

    plt.ylim(0, 1.1)
    plt.title(f'Model Metrics\nRun ID: {run_id}')
    plt.tight_layout()

    metric_file = f'metrics_{run_id}.png'
    plt.savefig(metric_file)
    plt.close()
    artifacts['metrics'] = metric_file

    return artifacts

def run_training_pipeline():
    """Proses pelatihan utama"""
    print("🚀 Memulai training model - Skilled CI")
    print("=" * 50)

    initialize_experiment()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"Train_Run_{run_id}"

    with mlflow.start_run(run_name=run_label):
        try:
            X_train, X_test, y_train, y_test = read_dataset()

            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'n_jobs': 1
            }

            print("🛠️ Melatih model RandomForest dengan konfigurasi:")
            print(params)

            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)

            print("🔁 Evaluasi via Cross Validation...")
            cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
            cv_avg = cv_scores.mean()
            cv_std = cv_scores.std()

            preds = clf.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted')
            rec = recall_score(y_test, preds, average='weighted')
            f1 = f1_score(y_test, preds, average='weighted')

            # Logging ke MLflow
            mlflow.log_params(params)
            mlflow.log_param("folds", 5)
            mlflow.log_param("run_id", run_id)

            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "cv_mean": cv_avg,
                "cv_std": cv_std
            })

            mlflow.sklearn.log_model(clf, "model", registered_model_name="SkilledModelClassifier")

            print("📎 Membuat artefak visualisasi...")
            artifacts = generate_artifacts(clf, X_train, X_test, y_test, preds, run_id)

            for name, path in artifacts.items():
                mlflow.log_artifact(path)
                print(f"✅ Artefak '{name}' berhasil diunggah")

            # Simpan ringkasan JSON
            summary = {
                "run": {
                    "id": run_id,
                    "name": run_label,
                    "time": datetime.now().isoformat()
                },
                "metrics": {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "cv_mean": cv_avg,
                    "cv_std": cv_std
                },
                "model_config": params,
                "features": list(X_train.columns),
                "importance": dict(zip(X_train.columns, clf.feature_importances_))
            }

            json_path = f"summary_{run_id}.json"
            with open(json_path, "w") as jf:
                json.dump(summary, jf, indent=2)
            mlflow.log_artifact(json_path)

            print("\n=== HASIL PELATIHAN ===")
            print(f"Akurasi: {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"CV: {cv_avg:.4f} ± {cv_std:.4f}")
            print(classification_report(y_test, preds, target_names=['1', '0']))

            # Cleanup file lokal
            for path in list(artifacts.values()) + [json_path]:
                if os.path.exists(path):
                    os.remove(path)

            return clf, acc, summary

        except Exception as err:
            print(f"❌ Terjadi kesalahan: {err}")
            mlflow.log_param("error", str(err))
            raise

if __name__ == "__main__":
    try:
        model, score, result = run_training_pipeline()
        print(f"\n🎉 Training selesai dengan akurasi: {score:.4f}")
        sys.exit(0)
    except Exception as error:
        print(f"\n💥 Proses gagal: {error}")
        sys.exit(1)
