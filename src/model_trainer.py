import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

class AnomalyDetector:
    def __init__(self, df, is_labeled=True):
        self.df = df
        self.is_labeled = is_labeled
        self.X = df[self.df.columns.difference(['label'])]  # keep only baseline features
        self.y = df["label"].copy() if is_labeled else None

    def train_isolation_forest(self):
        model = IsolationForest(n_estimators=100, contamination=0.5, random_state=42) # changing contamination will affect accuracy and other evaluation metrices
        model.fit(self.X)
        preds = model.predict(self.X)
        preds = np.where(preds == -1, 1, 0)
        return preds, model

    def train_dbscan(self, data=None):
        if data is None:
            data = self.df.copy()
        X = data.drop(columns=["label"], errors="ignore")
        print(f"DBSCAN training on data of shape: {X.shape}")

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        preds = dbscan.fit_predict(X)
        preds = [1 if p == -1 else 0 for p in preds]
        print(f"Length of predictions: {len(preds)}")
        return preds, dbscan
    

    def train_kmeans(self):
        model = KMeans(n_clusters=2, random_state=42)
        preds = model.fit_predict(self.X)
        # Use mean of cluster centers to map which one is anomaly
        center_dist = np.linalg.norm(model.cluster_centers_, axis=1)
        anomaly_cluster = np.argmax(center_dist)
        preds = np.where(preds == anomaly_cluster, 1, 0)
        return preds, model

    def evaluate_model(self, preds, model_name, true_labels = None):
        print(f"\nEvaluation for {model_name}:")
        '''true_label is added so that the DBSCAN won't face any issue as it work for some part of data 
        but other models like isolation forest and kmeans need full dataset
        '''
        y_true = None

        if true_labels is not None:
            print("Using provided true labels.")
            y_true = true_labels

        elif self.is_labeled:
            print("Using internal training labels.")
            y_true = self.y

        else:
            print("Unlabeled data — cannot compute evaluation metrics.")
            print("Anomaly prediction distribution:")
            print(pd.Series(preds).value_counts())
            return

        # Metrices for evaluation 
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, preds))

        print("\nClassification Report:")
        print(classification_report(y_true, preds, digits=4))

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        print(f"🔸Accuracy:  {acc:.4f}")
        print(f"🔸Precision: {prec:.4f}")
        print(f"🔸Recall:    {rec:.4f}")
        print(f"🔸F1 Score:  {f1:.4f}")

        try:
            roc_auc = roc_auc_score(y_true, preds)
            print(f"🔸ROC-AUC:   {roc_auc:.4f}")
        except ValueError:
            print("ROC-AUC:   Cannot be computed (single class in y_true or preds)")
