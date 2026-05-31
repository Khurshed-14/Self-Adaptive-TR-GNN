# Imports
## Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
from torch.utils.tensorboard import SummaryWriter 
import json
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.manifold import spectral_embedding
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import rbf_kernel
## Import Dataset Classes
from dataset_classes import ISO_NE, AT, SH_Dataset
## Import Model
from models_with_temporal_graph import TR_GNN_MultiScale
## Import Training and Testing Loops
from helper_functions import train_model, test_model, test_model_stepwise
# Train_Validation_Test
## ISO_NE
# Main Function
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ISO_NE(
        csv_path="data\iso_ne\selected_data_ISONE.csv",
        T_in=72,
        T_out=240,
        lag_hours=[1,12,24,168], 
        rolling_windows=[12,24],
    )

    total_len = len(dataset.df_numeric)
    train_split_idx = int(0.6 * total_len)
    val_split_idx = int(0.8 * total_len)
    
    print(f"Raw data length: {total_len}")
    print(f"Scaler 'train_size' (raw rows): {train_split_idx}")
    print(f"Scaler 'val_end' (raw rows): {val_split_idx}")

    scaler = StandardScaler()
    scaler.fit(dataset.df_numeric.iloc[:train_split_idx].values.astype(np.float32))
    
    print("\n🔍 Generating Feature Clusters...")
    dataset.apply_scaler(scaler)
    dataset.scaler = scaler  
    
    train_end = train_split_idx - dataset.T_in - dataset.T_out
    val_start = train_split_idx - dataset.T_in
    val_end = val_split_idx - dataset.T_in - dataset.T_out
    test_start = val_split_idx - dataset.T_in
    
    effective_len = len(dataset)
    train_end = min(train_end, effective_len)
    val_end = min(val_end, effective_len)

    train_idx = range(0, train_end)
    val_idx = range(val_start, val_end)
    test_idx = range(test_start, effective_len)

    print(f"Total valid samples: {effective_len}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    # DataLoaders can be reused since batch_size doesn't change in this sensitivity analysis
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    print(f"\n🚚 DataLoaders ready. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    hparams = {
        "N": dataset.N,
        "T_in": 72, # Input length in hours (3 days)
        "T_out": 240, # Output length in hours (10 days)
        "d": 32,
        "hidden_dim": 64, 
        "GCN_Layer": 5,
        "dropout_forecast": 0.1,
        "dropout_gcn": 0.2,
        "dropout_temporal": 0.2,
        "lr": 1e-3,
        "scheduler_patience": 3,
        "batch_size": 32, # Updated from 16
        "epochs": 100,
        "weight_decay": 1e-4, # Added this new hyperparameter
        "kernel_size": 7, # Added for temporal conv
        "dilation": 3, # Added for temporal conv
    }
    
    # --- Model ---
    model = TR_GNN_MultiScale(
        N=hparams["N"],
        T_in=hparams["T_in"],
        T_out=hparams["T_out"],
        d=hparams["d"],
        hidden_dim=hparams["hidden_dim"],
        GCN_Layer=hparams["GCN_Layer"],
        dropout_gcn=hparams["dropout_gcn"],
        dropout_temporal=hparams["dropout_temporal"],
        kernel_size=hparams["kernel_size"],
        dilation=hparams["dilation"],
    ).to(device)
    
    # Run name - change accordingly for each model
    run = f"TR_GNN_ISO_NE_Multi_Scale_GCN{hparams['GCN_Layer']}_Hidden{hparams['hidden_dim']}_Kernel{hparams['kernel_size']}_Dilation{hparams['dilation']}_LR{hparams['lr']}"
    
    log_dir = f"TR_GNN_ISO_NE/{run}"  # Define log directory for TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Log hyperparameters as text (avoid add_hparams which requires a metric_dict)
    writer.add_text("hparams", json.dumps(hparams, indent=2))
    
    print("\n🚀 Training TR-GNN_Multi_Scale model on ISO-NE dataset...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=hparams["epochs"],
        lr=hparams["lr"],
        device=device,
        scheduler_patience=hparams["scheduler_patience"],
        writer=writer,
        weight_decay=hparams["weight_decay"],
        save_path=f"Sens_Runs/{run}_best_model.pth"
    )

    print("\n🧪 Testing model performance...\n")
    preds, trues = test_model(
        dataset=dataset,
        model=model, test_loader=test_loader,
        device=device,
        writer=writer
    )
    
    writer.close()

if __name__ == "__main__":
    main()