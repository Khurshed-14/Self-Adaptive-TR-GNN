"""
Diagnostic script to verify graph learning fixes.
Tests:
1. Adjacency matrix evolution (should show diverse values, not all ~0)
2. W_q/W_k gradient flow (should be non-zero and changing)
3. Training convergence (loss should decrease)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from dataset_classes import ISO_NE
from models_with_temporal_graph import TR_GNN_MultiScale
from helper_functions import validate

def test_graph_learning():
    print("=" * 70)
    print("GRAPH LEARNING FIX DIAGNOSTIC TEST")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 Device: {device}")
    
    # --- Load dataset ---
    print("\n📂 Loading ISO-NE dataset...")
    dataset = ISO_NE(
        csv_path="data/iso_ne/selected_data_ISONE.csv",
        T_in=72,
        T_out=240,
        lag_hours=[1, 12, 24, 168],
        rolling_windows=[12, 24],
    )
    
    total_len = len(dataset.df_numeric)
    train_split_idx = int(0.6 * total_len)
    val_split_idx = int(0.8 * total_len)
    
    scaler = StandardScaler()
    scaler.fit(dataset.df_numeric.iloc[:train_split_idx].values.astype(np.float32))
    dataset.apply_scaler(scaler)
    dataset.scaler = scaler
    
    # Create small subset for quick test
    train_end = int(0.1 * (train_split_idx - dataset.T_in - dataset.T_out))  # 10% of train
    train_idx = range(0, train_end)
    val_idx = range(train_split_idx - dataset.T_in, min(train_split_idx - dataset.T_in + 100, val_split_idx - dataset.T_in - dataset.T_out))
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    
    print(f"   ✅ Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print(f"   ✅ Dataset features: {dataset.N}")
    
    # --- Initialize model ---
    print("\n🏗️  Initializing TR_GNN_MultiScale model...")
    model = TR_GNN_MultiScale(
        N=dataset.N,
        T_in=72,
        T_out=240,
        d=32,
        hidden_dim=64,
        GCN_Layer=5,
        dropout_gcn=0.2,
        dropout_temporal=0.2,
        kernel_size=7,
        dilation=3,
    ).to(device)
    
    print("   ✅ Model initialized")
    
    # --- Check initial W_q, W_k parameters ---
    print("\n🔍 Initial Parameter Check:")
    w_q_std = model.graph_learn.W_q.weight.data.std().item()
    w_k_std = model.graph_learn.W_k.weight.data.std().item()
    print(f"   W_q std: {w_q_std:.6f} (Xavier init should be ~{1/np.sqrt(64):.6f})")
    print(f"   W_k std: {w_k_std:.6f}")
    
    # --- Training loop with diagnostics ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    adjacency_matrices = []  # Store for visualization
    gradient_logs = []
    loss_logs = {"train": [], "val": []}
    
    print("\n🚀 Running 5 training epochs with diagnostics...\n")
    
    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        epoch_A = None
        
        for batch_idx, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            
            pred, A = model(X)
            
            # Store first batch adjacency matrix
            if batch_idx == 0:
                epoch_A = A.detach().cpu().numpy()[0]  # (N, N)
            
            mse_loss = criterion(pred, Y)
            smooth_loss = nn.functional.mse_loss(A[:-1], A[1:])
            sparse_loss = torch.norm(A, p=1)
            
            loss = mse_loss + 0.01 * smooth_loss + 1e-4 * sparse_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = validate(model, val_loader, criterion, device)
        
        # Log losses
        loss_logs["train"].append(avg_train_loss)
        loss_logs["val"].append(val_loss)
        
        # Log gradients
        w_q_grad = model.graph_learn.W_q.weight.grad.abs().mean().item() if model.graph_learn.W_q.weight.grad is not None else 0
        w_k_grad = model.graph_learn.W_k.weight.grad.abs().mean().item() if model.graph_learn.W_k.weight.grad is not None else 0
        gradient_logs.append({"w_q_grad": w_q_grad, "w_k_grad": w_k_grad})
        
        # Store adjacency matrix
        if epoch_A is not None:
            adjacency_matrices.append(epoch_A)
        
        print(f"Epoch {epoch}/5:")
        print(f"   Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"   W_q Grad Mean: {w_q_grad:.8f} | W_k Grad Mean: {w_k_grad:.8f}")
        print(f"   Adjacency Matrix Stats:")
        print(f"      - Min: {epoch_A.min():.6f}, Max: {epoch_A.max():.6f}, Mean: {epoch_A.mean():.6f}")
        print(f"      - Off-diagonal mean: {epoch_A[np.triu_indices_from(epoch_A, k=1)].mean():.6f}")
        print(f"      - Diagonal mean: {np.diag(epoch_A).mean():.6f}")
    
    # --- Visualization ---
    print("\n📊 Generating visualizations...\n")
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Loss curves
    ax1 = plt.subplot(2, 3, 1)
    epochs_range = range(1, 6)
    ax1.plot(epochs_range, loss_logs["train"], 'o-', label='Train Loss', linewidth=2, markersize=8)
    ax1.plot(epochs_range, loss_logs["val"], 's-', label='Val Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gradient evolution
    ax2 = plt.subplot(2, 3, 2)
    w_q_grads = [g["w_q_grad"] for g in gradient_logs]
    w_k_grads = [g["w_k_grad"] for g in gradient_logs]
    ax2.semilogy(epochs_range, w_q_grads, 'o-', label='W_q Gradient', linewidth=2, markersize=8)
    ax2.semilogy(epochs_range, w_k_grads, 's-', label='W_k Gradient', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean |Gradient|')
    ax2.set_title('Weight Gradients Over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3-5. Adjacency matrices at epochs 1, 3, 5
    for idx, epoch in enumerate([0, 2, 4]):
        ax = plt.subplot(2, 3, 4 + idx)
        A_epoch = adjacency_matrices[epoch]
        sns.heatmap(A_epoch, cmap='viridis', ax=ax, cbar=True, square=True, 
                    vmin=0, vmax=1, xticklabels=False, yticklabels=False)
        ax.set_title(f'Adjacency Matrix - Epoch {epoch + 1}')
    
    plt.tight_layout()
    plt.savefig('graph_learning_diagnostics.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: graph_learning_diagnostics.png")
    plt.close()
    
    # --- Summary Report ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    print("\n✅ CHECKS PASSED:")
    
    # Check 1: Loss decreased
    if loss_logs["val"][-1] < loss_logs["val"][0]:
        print(f"   ✓ Validation loss decreased: {loss_logs['val'][0]:.6f} → {loss_logs['val'][-1]:.6f}")
    else:
        print(f"   ✗ Validation loss did NOT decrease: {loss_logs['val'][0]:.6f} → {loss_logs['val'][-1]:.6f}")
    
    # Check 2: Gradients are non-zero
    if gradient_logs[-1]["w_q_grad"] > 1e-8 and gradient_logs[-1]["w_k_grad"] > 1e-8:
        print(f"   ✓ Gradients are flowing: W_q={gradient_logs[-1]['w_q_grad']:.8f}, W_k={gradient_logs[-1]['w_k_grad']:.8f}")
    else:
        print(f"   ✗ Gradients are NOT flowing properly")
    
    # Check 3: Adjacency matrix diversity
    A_final = adjacency_matrices[-1]
    off_diag_values = A_final[np.triu_indices_from(A_final, k=1)]
    off_diag_std = off_diag_values.std()
    if off_diag_std > 0.1:
        print(f"   ✓ Adjacency matrix shows diversity: off-diagonal std={off_diag_std:.6f}")
    else:
        print(f"   ✗ Adjacency matrix lacks diversity: off-diagonal std={off_diag_std:.6f}")
    
    # Check 4: Adjacency values NOT all near 0
    if A_final.mean() > 0.1:
        print(f"   ✓ Adjacency mean is healthy: {A_final.mean():.6f} (not collapsed to ~0)")
    else:
        print(f"   ✗ Adjacency values collapsed to near-zero: mean={A_final.mean():.6f}")
    
    # Check 5: Off-diagonal values exist
    if off_diag_values.max() > 0.3:
        print(f"   ✓ Off-diagonal edges learned: max={off_diag_values.max():.6f}")
    else:
        print(f"   ⚠ Off-diagonal edges weak: max={off_diag_values.max():.6f}")
    
    print("\n" + "=" * 70)
    print("Test complete! Check graph_learning_diagnostics.png for detailed analysis.")
    print("=" * 70)

if __name__ == "__main__":
    test_graph_learning()
