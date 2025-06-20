# CreditFraudSentinel: Hybrid GNN-LightGBM Fraud Detection System

This repository contains the source code for **CreditFraudSentinel**, an advanced fraud detection system that leverages a hybrid machine learning model to identify potentially fraudulent credit card transactions. The system combines the power of Graph Neural Networks (GNNs) for relational insights with a high-performance LightGBM model for classification.

![Fraud Network Visualization](fraud_network.png)

## 🚀 Key Features

- **Hybrid GNN + LightGBM Model**: Utilizes a GraphSAGE GNN to generate powerful relational embeddings from transaction data, which are then fed into a LightGBM classifier for robust fraud detection.
- **Dynamic Temporal Encoding**: Engineers features like Exponential Moving Averages (EMA) for transaction amounts and frequency to capture time-sensitive patterns and detect distributional shifts.
- **Graph-Based Feature Engineering**: Constructs a transaction graph to compute node centrality, adding a crucial network-aware feature to the model.
- **Adaptive SMOTE with Clustering**: Employs a sophisticated oversampling technique by first clustering fraud cases with DBSCAN, then applying SMOTE within each cluster. This preserves unique fraud patterns and creates more realistic synthetic data.
- **Insight Generation**: Includes a module to quickly generate feature importance and profile the characteristics of predicted fraudulent transactions, offering valuable insights for analysts.

## 📈 Results

After running the script, the following key outputs are generated:

- `roc_curve.html`: An interactive ROC curve plot to evaluate the model's performance.
- `fraud_network.png`: A visualization of the transaction network, highlighting detected fraud clusters.
- `fraud_detection.log`: A comprehensive log file detailing the entire execution pipeline, including model performance metrics and insights.

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- The `creditcard.csv` dataset is required. It can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/credit-card-fraud-detection).
- An NVIDIA GPU is recommended for accelerating the GNN embedding generation.

### Installation & Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hareeshkumarch/CreditFraudSentinel.git
    cd CreditFraudSentinel
    ```

2.  **Download the Dataset**: Download `creditcard.csv` and place it in the root of the project directory.

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU support, you may need to install a specific version of PyTorch. Please see the [PyTorch website](https://pytorch.org/) for instructions.*

4.  **Run the script:**
    ```bash
    python fraud_detection.py
    ```