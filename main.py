import os
import joblib
import numpy as np
import pandas as pd
from src.data_loader import KDDDataLoader
from src.model_trainer import AnomalyDetector
from src.visualizer import Visualizer

def main():
    print("\n*** Starting Anomaly Detection Pipeline ***")

    # === Load & Preprocess Training Data ===
    train_data_path = "data/raw/kddcup.data_10_percent_corrected"
    names_path = "data/raw/kddcup.names"
    
    loader = KDDDataLoader(train_data_path, names_path, is_labeled=True)
    train_df = loader.preprocess()
    baseline_features = loader.columns_to_keep # Store for reuse
    
    print("Training data loaded successfully. Unique labels:", train_df['label'].unique())

    # === Train Models ===
    print("------Training models-------")
    detector = AnomalyDetector(train_df, is_labeled=True)

    iso_preds, iso_model = detector.train_isolation_forest()
    km_preds, km_model = detector.train_kmeans()
    # for DBSACN use reduced dataset as can't work for large datasets
    sample_df = train_df.sample(n=10000, random_state=42).reset_index(drop=True)
    db_preds, db_model = detector.train_dbscan(data=sample_df)

    # === Evaluate Models ===
    print("------Evaluating models-------")
    detector.evaluate_model(iso_preds, "Isolation Forest")
    detector.evaluate_model(km_preds, "KMeans")
    detector.evaluate_model(db_preds, "DBSCAN", true_labels=sample_df['label'])

    # === Save Models ===
    print("-------Saving models using pickel for further use-----")
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(iso_model, "artifacts/isolation_forest.pkl")
    joblib.dump(db_model, "artifacts/dbscan.pkl")
    joblib.dump(km_model, "artifacts/kmeans.pkl")
    joblib.dump(loader.scaler, "artifacts/scaler.pkl")

    # === Load & Process Test Data ===
    print("-----Loading test data----")
    test_path = "data/raw/kddcup.testdata.unlabeled_10_percent"
    test_loader = KDDDataLoader(test_path, names_path, is_labeled=False, columns_to_keep=baseline_features)
    test_df = test_loader.preprocess()
    test_df = test_df[baseline_features]  # Align columns

    # === Visualization ===
    visual = Visualizer()
    # Visualize class distribution of training predictions
    visual.plot_class_distribution(iso_preds, title="Isolation Forest - Anomaly Distribution (Train)")
    visual.plot_class_distribution(km_preds, title="KMeans - Anomaly Distribution (Train)")
    visual.plot_class_distribution(db_preds, title="DBSCAN - Anomaly Distribution (Sampled Train)")

    # PCA plot of train vs test
    try:
        visual.plot_pca_distribution(train_df.drop(columns=['label']), test_df, title="PCA - Train vs Test")
    except Exception as e:
        print(f"## PCA visualization failed: {e}")

    # Confusion matrix
    try:
        visual.plot_confusion_matrix(train_df['label'], iso_preds, title="Confusion Matrix - Isolation Forest")
    except Exception as e:
        print(f"## Confusion matrix skipped: {e}")

    # Model comparison visualization
    try:
        print("====== Computing accuracy scores for model comparison ======")
        from sklearn.metrics import accuracy_score

        acc_iso = accuracy_score(train_df['label'], iso_preds)
        acc_km = accuracy_score(train_df['label'], km_preds)
        acc_db = accuracy_score(sample_df['label'], db_preds)  # DBSCAN was trained on sampled data

        visual.plot_model_comparison({
            "Isolation Forest": acc_iso,
            "KMeans": acc_km,
            "DBSCAN": acc_db
        })
    except Exception as e:
        print(f"## Model comparison skipped: {e}")


    print("\n*** Pipeline completed successfully. ***")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
