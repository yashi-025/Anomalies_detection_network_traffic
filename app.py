import streamlit as st
import pandas as pd
# import joblib
import os

from src.data_loader import KDDDataLoader
from src.model_trainer import AnomalyDetector
from src.visualizer import Visualizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("Modular Anomaly Detection Pipeline")

# ==== Load Training Data ====
st.header("Load and Preprocess Data")

default_train_path = "data/raw/kddcup.data_10_percent_corrected"
default_names_path = "data/raw/kddcup.names"

train_data_path = st.text_input("Path to KDD training data", default_train_path)
names_path = st.text_input("Path to KDD names file", default_names_path)

if st.button("Load Data"):
    loader = KDDDataLoader(train_data_path, names_path, is_labeled=True)
    train_df = loader.preprocess()
    baseline_features = loader.columns_to_keep
    st.session_state["train_df"] = train_df
    st.session_state["baseline_features"] = baseline_features
    st.success("Data Loaded")
    st.write("Sample Data", train_df.head())

# ==== Train Models ====
st.header("Train Models")

if "train_df" in st.session_state:
    train_df = st.session_state["train_df"]
    baseline_features = st.session_state["baseline_features"]
    detector = AnomalyDetector(train_df, is_labeled=True)

    selected_models = st.multiselect("Choose models to train", ["Isolation Forest", "KMeans", "DBSCAN"])

    if st.button("Train Selected Models"):
        st.session_state["models"] = {}
        st.session_state["preds"] = {}
        if "Isolation Forest" in selected_models:
            iso_preds, iso_model = detector.train_isolation_forest()
            st.session_state["models"]["Isolation Forest"] = iso_model
            st.session_state["preds"]["Isolation Forest"] = iso_preds
            st.success("Isolation Forest trained.")

        if "KMeans" in selected_models:
            km_preds, km_model = detector.train_kmeans()
            st.session_state["models"]["KMeans"] = km_model
            st.session_state["preds"]["KMeans"] = km_preds
            st.success("KMeans trained.")

        if "DBSCAN" in selected_models:
            sample_df = train_df.sample(n=10000, random_state=42).reset_index(drop=True)
            db_preds, db_model = detector.train_dbscan(data=sample_df)
            st.session_state["models"]["DBSCAN"] = db_model
            st.session_state["preds"]["DBSCAN"] = db_preds
            st.session_state["sample_df"] = sample_df
            st.success("DBSCAN trained on sample.")

# ==== Evaluate Models ====
st.header("Evaluate Models")

if "preds" in st.session_state and "train_df" in st.session_state:
    train_df = st.session_state["train_df"]
    y_true = train_df["label"]

    for name, preds in st.session_state["preds"].items():
        st.subheader(f"Evaluation: {name}")
        try:
            if name == "DBSCAN":
                sample_df = st.session_state["sample_df"]
                preds = preds[:len(sample_df)]  # ensure same length
                y_sample = sample_df["label"]
                acc = accuracy_score(y_sample, preds)
                prec = precision_score(y_sample, preds, zero_division=0)
                rec = recall_score(y_sample, preds, zero_division=0)
                f1 = f1_score(y_sample, preds, zero_division=0)
                cm = confusion_matrix(y_sample, preds)

                st.write(f"Accuracy: `{acc:.4f}`")
                st.write(f"Precision: `{prec:.4f}`")
                st.write(f"Recall: `{rec:.4f}`")
                st.write(f"F1 Score: `{f1:.4f}`")
                st.text("Confusion Matrix:")
                st.write(cm)

            else:
                acc = accuracy_score(y_true, preds)
                prec = precision_score(y_true, preds, zero_division=0)
                rec = recall_score(y_true, preds, zero_division=0)
                f1 = f1_score(y_true, preds, zero_division=0)
                cm = confusion_matrix(y_true, preds)

                st.write(f"Accuracy: `{acc:.4f}`")
                st.write(f"Precision: `{prec:.4f}`")
                st.write(f"Recall: `{rec:.4f}`")
                st.write(f"F1 Score: `{f1:.4f}`")
                st.text("Confusion Matrix:")
                st.write(cm)

        except Exception as e:
            st.warning(f"## Evaluation failed for {name}: {e}")

# ==== Visualizations ====
st.header("Visualizations")

if "preds" in st.session_state and "train_df" in st.session_state:
    train_df = st.session_state["train_df"]
    visual = Visualizer()

    with st.expander("Anomaly Class Distribution"):
        for name, preds in st.session_state["preds"].items():
            try:
                fig = visual.plot_class_distribution(preds, title=f"{name} - Anomaly Distribution")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"{name}: Distribution plot failed: {e}")

    with st.expander("PCA: Train vs Test"):
        try:
            test_path = "data/raw/kddcup.testdata.unlabeled_10_percent"
            test_loader = KDDDataLoader(test_path, names_path, is_labeled=False, columns_to_keep=st.session_state["baseline_features"])
            test_df = test_loader.preprocess()
            test_df = test_df[st.session_state["baseline_features"]]

            fig = visual.plot_pca_distribution(train_df.drop(columns=["label"]), test_df, title="PCA - Train vs Test")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"## PCA plot failed: {e}")

    with st.expander("Confusion Matrix"):
        try:
            if "Isolation Forest" in st.session_state["preds"]:
                iso_preds = st.session_state["preds"]["Isolation Forest"]
                fig = visual.plot_confusion_matrix(train_df["label"], iso_preds, title="Confusion Matrix - Isolation Forest")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"## Confusion matrix failed: {e}")

    with st.expander("Model Comparison (Accuracy)"):
        try:
            accs = {}
            if "Isolation Forest" in st.session_state["preds"]:
                accs["Isolation Forest"] = accuracy_score(train_df["label"], st.session_state["preds"]["Isolation Forest"])
            if "KMeans" in st.session_state["preds"]:
                accs["KMeans"] = accuracy_score(train_df["label"], st.session_state["preds"]["KMeans"])
            if "DBSCAN" in st.session_state["preds"]:
                sample_df = st.session_state["sample_df"]
                accs["DBSCAN"] = accuracy_score(sample_df["label"], st.session_state["preds"]["DBSCAN"])

            fig = visual.plot_model_comparison(accs)
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"## Model comparison failed: {e}")
