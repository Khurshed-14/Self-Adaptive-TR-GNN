"""
Full training on entire dataset with adjacency matrix visualization for all 15 epochs.
Displays adjacency matrices as heatmaps with values in each cell.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataset_classes import ISO_NE
from models_with_temporal_graph import TR_GNN_MultiScale
from helper_functions import test_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('adjacency_matrices', exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

print("="*80)
print("FULL TRAINING WITH GRAPH LEARNING VISUALIZATION - 100 EPOCHS")
print("="*80)

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\n📂 Loading ISO-NE dataset (FULL)...")
dataset = ISO_NE(
    csv_path="data/iso_ne/selected_data_ISONE.csv",
    T_in=72,
    T_out=240,
    lag_hours=[1, 12, 24, 168],
    rolling_windows=[12, 24],
)

# Apply scaler
total_len = len(dataset.df_numeric)
train_split_idx = int(0.6 * total_len)
val_split_idx = int(0.8 * total_len)

print(f"   Raw data length: {total_len}")

scaler = StandardScaler()
scaler.fit(dataset.df_numeric.iloc[:train_split_idx].values.astype(np.float32))
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

print(f"   ✅ Dataset loaded: {effective_len} valid samples (19 features)")
print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

train_subset = Subset(dataset, train_idx)
val_subset = Subset(dataset, val_idx)
test_subset = Subset(dataset, test_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================================
# INITIALIZE MODEL
# ============================================================================
print("\n🏗️  Initializing TR_GNN_MultiScale model...")
model = TR_GNN_MultiScale(
    N=19,  # Number of features
    T_in=72,
    T_out=240,
    d=32,
    hidden_dim=64,
    dropout_temporal=0.2,
    dropout_gcn=0.2,
    dropout_forecast=0.3,
    GCN_Layer=5,
    kernel_size=7,
    dilation=3,
    graph_temperature=3.0
).to(DEVICE)
print("   ✅ Model initialized")

# ============================================================================
# TRAINING SETUP
# ============================================================================
# ============================================================================
# OPTIMIZER WITH DIFFERENTIAL LEARNING RATES
# ============================================================================
# Graph learning parameters need higher learning rate to overcome local optima
graph_param_ids = set(id(p) for p in model.graph_learn.parameters())
param_groups = [
    {'params': [p for p in model.parameters() if id(p) in graph_param_ids], 'lr': LEARNING_RATE * 5.0},
    {'params': [p for p in model.parameters() if id(p) not in graph_param_ids], 'lr': LEARNING_RATE}
]

optimizer = torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)

adjacency_matrices = []  # Store one adjacency matrix per epoch
train_losses = []
val_losses = []

# Early stopping setup
best_val_loss = float('inf')
epochs_no_improve = 0
early_stopping_patience = 10

# Create directory to save adjacency matrices
import os
os.makedirs('adjacency_matrices', exist_ok=True)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n🚀 Running 100 training epochs...\n")

for epoch in range(1, EPOCHS + 1):
    # ---- TRAIN ----
    model.train()
    epoch_train_loss = 0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)  # (B, 72, N), (B, 240, N)
        
        optimizer.zero_grad()
        
        # Forward pass
        Y_pred, A = model(X)  # Y_pred: (B, 240, N), A: (B, N, N)
        
        # Pure prediction loss (graph learning happens through gradient flow)
        loss = nn.MSELoss()(Y_pred, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_train_loss += loss.item()
    
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)
    
    # ---- VALIDATION ----
    model.eval()
    epoch_val_loss = 0
    epoch_A = None
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            Y_pred, A = model(X)
            
            # Pure prediction loss for monitoring
            loss = nn.MSELoss()(Y_pred, y)
            
            epoch_val_loss += loss.item()
            
            # Store first batch's adjacency matrix for visualization
            if epoch_A is None:
                epoch_A = A[0].detach().cpu().numpy()  # (N, N)
    
    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)
    
    scheduler.step(epoch_val_loss)
    
    # Store adjacency matrix
    if epoch_A is not None:
        adjacency_matrices.append(epoch_A)
        
        # Save adjacency matrix as numpy file
        np.save(f'adjacency_matrices/epoch{epoch}.npy', epoch_A)
        
        # Save adjacency matrix as image
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(epoch_A, cmap='YlOrRd', vmin=0, vmax=0.8, aspect='auto')
        
        # Add values to cells
        N = epoch_A.shape[0]
        for i in range(N):
            for j in range(N):
                value = epoch_A[i, j]
                text_color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=5, fontweight='bold')
        
        ax.set_title(f'Adjacency Matrix - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label='Adjacency Weight')
        plt.tight_layout()
        plt.savefig(f'adjacency_matrices/epoch{epoch}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    # Early stopping check
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        improve_status = "✅ IMPROVED"
    else:
        epochs_no_improve += 1
        improve_status = f"❌ No improve ({epochs_no_improve}/{early_stopping_patience})"
    
    # Print progress
    print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | {improve_status}")
    print(f"           Adj Matrix Mean: {epoch_A.mean():.4f}, Std: {epoch_A.std():.4f}")
    print(f"           Off-diag Mean: {(epoch_A - np.eye(epoch_A.shape[0])).mean():.4f}")
    print()
    
    # Early stopping
    if epochs_no_improve >= early_stopping_patience:
        print(f"⛔ Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
        break

# ============================================================================
# VISUALIZATION: ADJACENCY MATRICES WITH VALUES
# ============================================================================
print("📊 Generating comprehensive visualization...")

# Create adaptive grid based on actual number of epochs
num_epochs_trained = len(adjacency_matrices)
num_cols = min(10, (num_epochs_trained + 4) // 5)  # Dynamic columns
num_rows = (num_epochs_trained + num_cols - 1) // num_cols  # Dynamic rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(min(28, 3*num_cols), min(20, 3*num_rows)))
fig.suptitle(f'Adjacency Matrix Evolution Over {num_epochs_trained} Epochs\n(Values shown inside cells)', 
             fontsize=18, fontweight='bold')

axes = axes.flatten() if hasattr(axes, 'flatten') else [axes] if num_rows == 1 and num_cols == 1 else axes.flatten()
vmin, vmax = 0, 0.8

for epoch_idx, A in enumerate(adjacency_matrices):
    ax = axes[epoch_idx]
    
    # Create heatmap
    im = ax.imshow(A, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add text annotations with values for ALL cells
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            value = A[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            # Show all values with small font
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   color=text_color, fontsize=3, fontweight='bold')
    
    # Labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Epoch {epoch_idx + 1:2d}', fontsize=12, fontweight='bold')
    
    # Add statistics to title
    mean_val = A.mean()
    off_diag = (A - np.eye(N)).mean()
    ax.text(0.5, -0.05, f'μ={mean_val:.3f}, off-diag={off_diag:.3f}', 
            transform=ax.transAxes, ha='center', fontsize=9, style='italic')

# Hide unused subplots
for epoch_idx in range(len(adjacency_matrices), len(axes)):
    axes[epoch_idx].set_visible(False)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Adjacency Weight', fontsize=12)

plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.savefig('adjacency_evolution_100epochs.png', dpi=200, bbox_inches='tight')
print("   ✅ Saved: adjacency_evolution_100epochs.png")
plt.close()

# ============================================================================
# VISUALIZATION: TRAINING CURVES
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
actual_epochs = len(train_losses)
axes[0].plot(range(1, actual_epochs + 1), train_losses, 'o-', label='Train Loss', linewidth=2, markersize=6)
axes[0].plot(range(1, actual_epochs + 1), val_losses, 's-', label='Val Loss', linewidth=2, markersize=6)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title(f'Training Curves - {actual_epochs} Epochs (Early Stopped)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Adjacency matrix statistics over epochs
adj_means = [A.mean() for A in adjacency_matrices]
adj_stds = [A.std() for A in adjacency_matrices]
off_diag_means = [(A - np.eye(A.shape[0])).mean() for A in adjacency_matrices]

axes[1].plot(range(1, actual_epochs + 1), adj_means, 'o-', label='Mean', linewidth=2, markersize=6)
axes[1].plot(range(1, actual_epochs + 1), adj_stds, 's-', label='Std Dev', linewidth=2, markersize=6)
axes[1].plot(range(1, actual_epochs + 1), off_diag_means, '^-', label='Off-diag Mean', linewidth=2, markersize=6)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Value', fontsize=12)
axes[1].set_title('Adjacency Matrix Statistics Over Epochs', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_statistics_100epochs.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: training_statistics_100epochs.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
actual_epochs = len(train_losses)
print(f"\n📊 Final Results (Epoch {actual_epochs}):")
print(f"   Train Loss: {train_losses[-1]:.6f}")
print(f"   Val Loss:   {val_losses[-1]:.6f}")
print(f"\n📈 Adjacency Matrix Statistics:")
print(f"   Initial Mean:      {adjacency_matrices[0].mean():.4f}")
print(f"   Final Mean:        {adjacency_matrices[-1].mean():.4f}")
print(f"   Initial Std:       {adjacency_matrices[0].std():.4f}")
print(f"   Final Std:         {adjacency_matrices[-1].std():.4f}")
print(f"\n🎯 Off-diagonal Connections:")
print(f"   Initial Off-diag:  {(adjacency_matrices[0] - np.eye(adjacency_matrices[0].shape[0])).mean():.4f}")
print(f"   Final Off-diag:    {(adjacency_matrices[-1] - np.eye(adjacency_matrices[-1].shape[0])).mean():.4f}")
print("\n✅ Visualizations & Adjacency Matrices saved!")
print("   - adjacency_evolution_100epochs.png (heatmaps with all values displayed)")
print("   - training_statistics_100epochs.png (loss curves & adjacency stats)")
print(f"   - adjacency_matrices/ directory with epoch1.png, epoch1.npy, epoch2.png, epoch2.npy, ... (individual epochs)")
print("="*80)

# ============================================================================
# TEST LOOP
# ============================================================================
print("\n" + "="*80)
print("🧪 TESTING ON HELD-OUT TEST SET")
print("="*80)

model.eval()
all_preds = []
all_trues = []
test_loss = 0.0

print("\n📊 Evaluating on test set...")
with torch.no_grad():
    for X_batch, y_batch in DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        # Forward pass
        y_pred, _ = model(X_batch)
        
        # Calculate loss
        loss = nn.MSELoss()(y_pred, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        
        # Store predictions and targets
        all_preds.append(y_pred.cpu().numpy())
        all_trues.append(y_batch.cpu().numpy())

# Concatenate all batches
all_preds = np.concatenate(all_preds, axis=0)  # (num_samples, T_out, 1) or (num_samples, T_out)
all_trues = np.concatenate(all_trues, axis=0)

test_loss /= len(test_subset)

# Calculate metrics
mse = mean_squared_error(all_trues.flatten(), all_preds.flatten())
mae = mean_absolute_error(all_trues.flatten(), all_preds.flatten())
rmse = np.sqrt(mse)
r2 = r2_score(all_trues.flatten(), all_preds.flatten())

print(f"\n📈 Test Set Results:")
print(f"   Test Loss (MSE):  {test_loss:.6f}")
print(f"   Test RMSE:        {rmse:.6f}")
print(f"   Test MAE:         {mae:.6f}")
print(f"   Test R² Score:    {r2:.6f}")

print("\n" + "="*80)
print("✅ TESTING COMPLETE")
print("="*80)
