# Anomalies Detection in Network Traffic using Unsupervised Learning

A modular and interactive machine learning pipeline for detecting anomalies in network traffic using unsupervised learning techniques. Designed for robust detection, monitoring on the **KDD Cup 1999 dataset**.

## Overview

Network anomaly detection is critical for identifying security breaches, network failures, and suspicious activity. This project leverages unsupervised machine learning techniques to:
- Detect unusual network patterns.
- Evaluate multiple models for comparative performance.

## Features

- **Data Preprocessing** (categorical encoding, feature selection)
- **Unsupervised Models**: Isolation Forest, KMeans, DBSCAN
- **Evaluation Metrics**: Accuracy, Confusion Matrix, F1-Score, Precision, Recall
- **Visualizer**: Class distribution, t-SNE, PCA, model comparison
- **Streamlit UI** for model training, prediction, and feedback

## 📁 Project Structure

```bash
.
├── data/                      # KDD Cup 1999 dataset files
├── src/                      
│   ├── model_trainer.py         # Isolation Forest, KMeans, DBSCAN and evaluation metrices
│   ├── data_loader.py           # For preprocessing, normalization
│   └── visualizer.py            # Visualizations (confusion matrix, t-SNE, PCA, etc.)
├── artifacts/                   # conatin .pkl files 
│   ├── isolation_forest.pkl        
│   ├── kmeans.pkl          
│   ├── dbscan.pkl
|   └──scaler.pkl            
├── main.py                    # Main pipeline for training and evaluation
├── streamlit_app.py           # Interactive app using Streamlit
├── requirements.txt           
└── README.md                  
