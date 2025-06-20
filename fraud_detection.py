import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, roc_curve, auc
from scipy.stats import ks_2samp
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# 1. Setup Logging
def setup_logging():
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("fraud_detection.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")

# 2. Data Loading and Preprocessing
def load_and_preprocess_data(file_path, nrows=None):
    """Loads, preprocesses, and splits the dataset."""
    try:
        logging.info(f"Loading data from {file_path} with {nrows} rows.")
        df = pd.read_csv(file_path, nrows=nrows)
        
        # Drop Time as it's not directly useful, we'll use its properties for temporal features
        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)

        # Handle potential missing values (though this dataset is clean)
        if df.isnull().sum().any():
            logging.warning("Missing values found. Filling with median.")
            df = df.fillna(df.median())

        X = df.drop('Class', axis=1)
        y = df['Class']

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        logging.info("Data loading and preprocessing complete.")
        return df, X, y
    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} was not found.")
        return None, None, None

# 3. Dynamic Temporal Encoding
def dynamic_temporal_encoding(df):
    """Engineers dynamic temporal features."""
    logging.info("Starting dynamic temporal encoding.")
    # Sort by a proxy for time if 'Time' was dropped but index reflects order
    df_sorted = df.sort_index()

    # EMA for transaction amount
    df_sorted['ema_amount'] = df_sorted['Amount'].ewm(span=100).mean()

    # EMA for transaction frequency (using counts over a window)
    df_sorted['ema_freq'] = df_sorted.index.to_series().diff().ewm(span=100).mean()
    df_sorted['ema_freq'] = df_sorted['ema_freq'].fillna(0) # Fill NaN for the first element
    
    # Kolmogorov-Smirnov test for distributional shifts
    # We split data in half to check for a shift
    split_point = len(df_sorted) // 2
    part1 = df_sorted['Amount'][:split_point]
    part2 = df_sorted['Amount'][split_point:]
    ks_stat, p_value = ks_2samp(part1, part2)
    logging.info(f"KS test for Amount distribution shift: Statistic={ks_stat:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        logging.warning("Significant distribution shift detected in 'Amount'.")

    logging.info("Dynamic temporal encoding complete.")
    return df_sorted[['ema_amount', 'ema_freq']]

# 4. Graph Construction and GNN
def construct_graph_and_generate_embeddings(df):
    """Constructs a transaction graph and generates GNN embeddings."""
    logging.info("Constructing graph and generating GNN embeddings.")
    
    # Create a simple graph where transactions are nodes
    G = nx.Graph()
    G.add_nodes_from(df.index)

    # Create synthetic edges: connect transactions close to each other
    # This is a proxy for related activity
    for i in range(len(df) - 1):
        # Connect each transaction to the next one
        G.add_edge(i, i + 1)
        # Connect transactions with similar amounts
        if abs(df['Amount'][i] - df['Amount'][i+1]) < 0.1: # Threshold for similarity
             G.add_edge(i, i+1)


    # Compute centrality as a feature
    centrality = nx.degree_centrality(G)
    centrality_feature = pd.Series(centrality, name='centrality').fillna(0)

    # Prepare data for PyTorch Geometric
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(df.drop('Class', axis=1).values, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index)

    # Define GraphSAGE model
    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GraphSAGE, self).__init__()
            self.sage1 = SAGEConv(in_channels, 32)
            self.sage2 = SAGEConv(32, out_channels)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.sage1(x, edge_index)
            x = F.relu(x)
            x = self.sage2(x, edge_index)
            return x

    # Generate embeddings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(graph_data.num_node_features, 16).to(device)
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        embeddings = model(graph_data).cpu().numpy()

    logging.info("Graph construction and GNN embeddings generation complete.")
    return embeddings, centrality_feature

# 5. Adaptive SMOTE with Clustering
def adaptive_smote_with_clustering(X, y):
    """Applies adaptive SMOTE based on fraud clusters."""
    logging.info("Applying adaptive SMOTE with clustering.")
    
    X_fraud = X[y == 1]
    
    # If no fraud cases, can't do SMOTE
    if len(X_fraud) == 0:
        logging.warning("No fraud cases found. Skipping SMOTE.")
        return X, y
    
    # Use DBSCAN to find clusters of fraud
    # eps needs tuning; smaller for denser data
    db = DBSCAN(eps=0.5, min_samples=3).fit(X_fraud)
    labels = db.labels_
    
    X_resampled_list, y_resampled_list = [X], [y]
    
    # Apply SMOTE to each cluster
    for cluster_label in np.unique(labels):
        if cluster_label == -1: continue # Skip noise points
        
        cluster_indices = X_fraud.index[labels == cluster_label]
        X_cluster = X.loc[cluster_indices]
        y_cluster = y.loc[cluster_indices]
        
        # We need some non-fraud samples to give to SMOTE
        X_combined = pd.concat([X[y == 0], X_cluster])
        y_combined = pd.concat([y[y == 0], y_cluster])

        # Apply SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_res, y_res = smote.fit_resample(X_combined, y_combined)

        # Keep only the newly generated fraud samples and original data
        X_resampled_list.append(X_res[len(X_combined):])
        y_resampled_list.append(y_res[len(X_combined):])

    X_resampled = pd.concat(X_resampled_list)
    y_resampled = pd.concat(y_resampled_list)

    logging.info(f"Adaptive SMOTE complete. New dataset size: {len(y_resampled)}, fraud cases: {y_resampled.sum()}")
    return X_resampled, y_resampled

# 6. Hybrid Model Training and Evaluation
def train_and_evaluate_lightgbm(X_train, y_train, X_test, y_test):
    """Trains and evaluates the LightGBM model."""
    logging.info("Training and evaluating LightGBM model.")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 1000,
        'is_unbalance': True
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
    
    return model, y_pred, y_pred_proba

# 7. Visualization: ROC Curve
def plot_roc_curve(y_test, y_pred_proba):
    """Plots and saves the ROC curve."""
    logging.info("Plotting ROC curve.")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                             mode='lines',
                             name=f'ROC curve (area = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             line=dict(dash='dash'),
                             name='Chance'))
    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      legend=dict(x=0, y=1, traceorder='reversed'))
    
    fig.write_html("roc_curve.html")
    logging.info("ROC curve saved to roc_curve.html")

# 8. Generate Insights
def generate_and_log_insights(model, X_test, y_test, y_pred, df):
    """Generates and logs insights from the model's predictions."""
    logging.info("--- Generating Model Insights ---")

    # 1. Feature Importance from LightGBM
    feature_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    logging.info("Top 10 Most Important Features:")
    logging.info("\n" + feature_importances.head(10).to_string())

    # 2. Profile of Predicted Frauds vs. Actual Frauds
    X_analysis = X_test.copy()
    # Get unscaled 'Amount' for interpretability
    X_analysis['Amount'] = df['Amount'].loc[X_test.index]
    X_analysis['actual_class'] = y_test
    X_analysis['predicted_class'] = y_pred

    key_features = ['Amount', 'centrality', 'ema_amount', 'ema_freq']
    
    predicted_fraud_profile = X_analysis[X_analysis['predicted_class'] == 1][key_features].describe()
    actual_fraud_profile = X_analysis[X_analysis['actual_class'] == 1][key_features].describe()

    logging.info("\n--- Profile of PREDICTED Frauds ---")
    logging.info(predicted_fraud_profile)

    logging.info("\n--- Profile of ACTUAL Frauds ---")
    logging.info(actual_fraud_profile)

    logging.info("\n--- Key Insight Takeaways ---")
    logging.info("1. The model's feature importance list shows which transaction attributes are most indicative of fraud.")
    predicted_fraud_mean_amount = predicted_fraud_profile.loc['mean', 'Amount']
    actual_fraud_mean_amount = actual_fraud_profile.loc['mean', 'Amount']
    logging.info(f"2. The average 'Amount' for transactions the model flags as fraud is ${predicted_fraud_mean_amount:.2f}, compared to the average for actual frauds (${actual_fraud_mean_amount:.2f}).")
    logging.info("3. By comparing these profiles, an analyst can understand the model's behavior and the nature of false positives.")

# 9. Visualization: Fraud Network
def visualize_fraud_network(df, y):
    """Visualizes the fraud transaction network."""
    logging.info("Visualizing fraud network.")
    
    fraud_indices = df.index[y == 1]
    if len(fraud_indices) == 0:
        logging.warning("No fraud nodes to visualize.")
        return

    G = nx.Graph()
    G.add_nodes_from(df.index)
    
    for i in range(len(df) - 1):
        G.add_edge(i, i + 1)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)
    
    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, nodelist=list(df.index), node_color='lightblue', node_size=50)
    # Highlight fraud nodes
    nx.draw_networkx_nodes(G, pos, nodelist=fraud_indices, node_color='red', node_size=100)
    
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    
    plt.title("Transaction Network with Fraud Highlighting")
    plt.savefig("fraud_network.png")
    plt.close()
    logging.info("Fraud network visualization saved to fraud_network.png")


# Main execution block
def main():
    """Main function to run the fraud detection pipeline."""
    setup_logging()
    
    # Use the provided dataset name and row limit
    df, X, y = load_and_preprocess_data('creditcard.csv', nrows=100000)
    if df is None:
        return
    
    temporal_features = dynamic_temporal_encoding(df)
    embeddings, centrality = construct_graph_and_generate_embeddings(df)

    # Combine all features
    X_full = X.copy()
    X_full['centrality'] = centrality
    X_full[['ema_amount', 'ema_freq']] = temporal_features
    
    # GNN embeddings as a DataFrame
    embedding_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols, index=X_full.index)
    X_full = pd.concat([X_full, embeddings_df], axis=1)
    
    logging.info(f"Full feature set created with shape: {X_full.shape}")

    # Split data before resampling
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.3, random_state=42, stratify=y)
    
    # Apply adaptive SMOTE only on the training data
    X_train_resampled, y_train_resampled = adaptive_smote_with_clustering(X_train, y_train)
    
    # Train and evaluate
    model, y_pred, y_pred_proba = train_and_evaluate_lightgbm(X_train_resampled, y_train_resampled, X_test, y_test)
    
    # Generate Insights
    generate_and_log_insights(model, X_test, y_test, y_pred, df)

    # Visualizations
    plot_roc_curve(y_test, y_pred_proba)
    visualize_fraud_network(df, y) # Visualize on original data distribution

    logging.info("Fraud detection script finished successfully.")


if __name__ == '__main__':
    main() 