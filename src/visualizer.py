import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import streamlit as st

sns.set(style="whitegrid")

class Visualizer:

    @staticmethod
    def plot_class_distribution(y, title="Class Distribution", labels={0: "Normal", 1: "Anomaly"}):
        """
        Visualize class distribution in the dataset.
        """
        y_mapped = pd.Series(y).map(labels)
        plt.figure()
        ax = sns.countplot(x=y_mapped, palette="Set2")
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

    @staticmethod
    def plot_tsne_projection(X, y=None, title="t-SNE Projection (Feature Space)", sample_size=1000):
        """
        Visualize high-dimensional data in 2D using t-SNE.
        """
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_sample = X[:sample_size]
        y_sample = y[:sample_size] if y is not None else None

        X_proj = tsne.fit_transform(X_sample)
        plt.figure(figsize=(8, 6))
        if y_sample is not None:
            sns.scatterplot(x=X_proj[:, 0], y=X_proj[:, 1], hue=y_sample, palette="coolwarm", alpha=0.7)
            plt.legend(title="Class")
        else:
            plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.6)
        plt.title(title)
        plt.tight_layout()
        st.pyplot(plt)

    @staticmethod
    def plot_prediction_distribution(preds, title="Model Prediction Distribution", labels={0: "Normal", 1: "Anomaly"}):
        """
        Visualize the distribution of model predictions.
        """
        preds_mapped = pd.Series(preds).map(labels)
        plt.figure()
        ax = sns.countplot(x=preds_mapped, palette="Set1")
        ax.set_title(title)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

    def plot_pca_distribution(self, train_data, test_data, title="PCA Projection of Train vs Test"):
        # Combine data
        combined = pd.concat([
            pd.DataFrame(train_data).assign(dataset='Train'),
            pd.DataFrame(test_data).assign(dataset='Test')
        ])

        # Drop label columns if any exist
        if 'label' in combined.columns:
            combined = combined.drop(columns=['label'])

        # Fit PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined.drop(columns=['dataset']))

        # Prepare DataFrame for plotting
        reduced_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        reduced_df['Dataset'] = combined['dataset'].values

        plt.figure(figsize=(8, 6))
        for dataset in ['Train', 'Test']:
            subset = reduced_df[reduced_df['Dataset'] == dataset]
            plt.scatter(subset["PC1"], subset["PC2"], label=dataset, alpha=0.5)

        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels={0: "Normal", 1: "Anomaly"}, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[labels[0], labels[1]])
        fig, ax = plt.subplots()
        disp.plot(cmap="Blues", ax=ax)
        plt.title(title)
        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def plot_model_comparison(model_scores, title="Model Accuracy Comparison"):
        """
        model_scores: dict with keys as model names and values as accuracy or F1-score
        Example: {'Isolation Forest': 0.67, 'KMeans': 0.58, 'DBSCAN': 0.45}
        """
        model_names = list(model_scores.keys())
        scores = list(model_scores.values())

        plt.figure(figsize=(8, 5))
        sns.barplot(x=model_names, y=scores, palette="Set3")
        plt.title(title)
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
