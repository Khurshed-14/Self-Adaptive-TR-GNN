"""
Full training with covariate -> demand graph structure (target_in mask).

Learns which features drive demand: only self-loops and edges into the target
node are allowed. Saves labeled adjacency heatmaps on validation improvement only.

Outputs go to adjacency_matrices_demand/ (does not overwrite adjacency_matrices/).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from dataset_classes import AT
from models_with_temporal_graph import TR_GNN_MultiScale
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

ADJ_OUT_DIR = 'adjacency_matrices_demand'
EVOLUTION_PNG = 'adjacency_evolution_demand.png'
STATS_PNG = 'training_statistics_demand.png'


def _shorten_labels(names, max_len=14):
    out = []
    for name in names:
        s = str(name)
        out.append(s if len(s) <= max_len else s[: max_len - 1] + "\u2026")
    return out


def _save_adjacency_epoch(epoch_A, epoch, feature_names, target_name, out_dir=ADJ_OUT_DIR, vmin=0.0, vmax=0.8):
    """Save .npy and labeled heatmap(s) for one checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'epoch{epoch}.npy'), epoch_A)

    num_heads, n, _ = epoch_A.shape
    labels = _shorten_labels(feature_names)
    side = max(6.0, n * 0.35)
    fig, axes = plt.subplots(1, num_heads, figsize=(side * num_heads + 0.8, side + 0.6))
    if num_heads == 1:
        axes = [axes]

    mid = (vmin + vmax) * 0.5
    cell_fs = max(3.5, min(7.0, 90 / n))
    last_im = None
    for h, ax in enumerate(axes):
        mat = epoch_A[h]
        last_im = ax.imshow(mat, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'Head {h + 1}' if num_heads > 1 else 'Adjacency', fontsize=10, fontweight='bold')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel('From feature (j)', fontsize=7)
        if h == 0:
            ax.set_ylabel('To feature (i)', fontsize=7)
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                color = 'white' if val > mid else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=cell_fs, fontweight='bold')

    fig.suptitle(
        f'What Drives {target_name} — Epoch {epoch}\n(row = receiver, col = sender)',
        fontsize=13, fontweight='bold',
    )
    fig.colorbar(last_im, ax=axes, label='Adjacency Weight', shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'epoch{epoch}.png'), dpi=100, bbox_inches='tight')
    plt.close()


os.makedirs(ADJ_OUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAPH_STRUCTURE = 'target_in'
L2_ALPHA = 0.0005

print("="*80)
print("FULL TRAINING — WHAT DRIVES DEMAND (target_in graph)")
print("="*80)

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\n📂 Loading AT dataset (FULL)...")
dataset = AT(
    csv_path="data/at/AT Dataset.csv",
    T_in=72,
    T_out=240,
    lag_hours=[1, 12, 24, 168],
    rolling_windows=[12, 24],
)

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

feature_names = dataset.feature_names
target_name = feature_names[dataset.target_idx]
print(f"   ✅ Dataset loaded: {effective_len} valid samples ({dataset.N} features)")
print(f"   Target: {target_name} (index {dataset.target_idx})")
print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

train_subset = Subset(dataset, train_idx)
val_subset = Subset(dataset, val_idx)
test_subset = Subset(dataset, test_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================================
# INITIALIZE MODEL
# ============================================================================
print("\n🏗️  Initializing TR_GNN_MultiScale model...")
model = TR_GNN_MultiScale(
    N=dataset.N,
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
    graph_temperature=1.0,
    target_idx=dataset.target_idx,
    graph_structure=GRAPH_STRUCTURE,
).to(DEVICE)
print(f"   ✅ Model initialized (graph_structure={GRAPH_STRUCTURE})")

# ============================================================================
# TRAINING SETUP
# ============================================================================
graph_param_ids = set(id(p) for p in model.graph_learn.parameters())
param_groups = [
    {'params': [p for p in model.parameters() if id(p) in graph_param_ids],
     'lr': LEARNING_RATE, 'weight_decay': 0.0},
    {'params': [p for p in model.parameters() if id(p) not in graph_param_ids],
     'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
]
optimizer = torch.optim.Adam(param_groups)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)

adjacency_matrices = []
improved_epoch_nums = []
train_losses = []
val_losses = []

best_val_loss = float('inf')
epochs_no_improve = 0
early_stopping_patience = 10

# ============================================================================
# TRAINING LOOP
# ============================================================================
print(f"\n🚀 Running {EPOCHS} training epochs...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_train_loss = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        Y_pred, A = model(X)
        mse_loss = nn.MSELoss()(Y_pred, y)

        N = A.size(-1)
        identity_mask = torch.eye(N, device=DEVICE)
        off_diag_mask = 1.0 - identity_mask
        off_diag_mask[dataset.target_idx, :] = 0.0
        l2_reg = torch.mean((A * off_diag_mask) ** 2)

        loss = mse_loss + (L2_ALPHA * l2_reg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_train_loss += mse_loss.item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    epoch_val_loss = 0
    epoch_A = None

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            Y_pred, A = model(X)
            epoch_val_loss += nn.MSELoss()(Y_pred, y).item()
            if epoch_A is None:
                epoch_A = np.expand_dims(A.detach().cpu().numpy(), axis=0)

    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)
    scheduler.step(epoch_val_loss)

    saved_adj_this_epoch = False
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        improve_status = "✅ IMPROVED"
        if epoch_A is not None:
            adjacency_matrices.append(epoch_A)
            improved_epoch_nums.append(epoch)
            _save_adjacency_epoch(epoch_A, epoch, feature_names, target_name)
            saved_adj_this_epoch = True
    else:
        epochs_no_improve += 1
        improve_status = f"❌ No improve ({epochs_no_improve}/{early_stopping_patience})"

    print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | {improve_status}")
    if epoch_A is not None:
        n_nodes = epoch_A.shape[-1]
        demand_row = epoch_A[0, dataset.target_idx, :]
        demand_row_off = demand_row.copy()
        demand_row_off[dataset.target_idx] = 0
        print(f"           Adj Matrix Mean: {epoch_A.mean():.4f}, Std: {epoch_A.std():.4f}")
        print(f"           {target_name} row (drivers) off-diag sum: {demand_row_off.sum():.4f}")
        if saved_adj_this_epoch:
            print(f"           💾 Saved checkpoint -> {ADJ_OUT_DIR}/epoch{epoch}.npy / .png")
    print()

    if epochs_no_improve >= early_stopping_patience:
        print(f"⛔ Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
        break

# ============================================================================
# VISUALIZATION
# ============================================================================
if adjacency_matrices:
    print("📊 Generating demand-driver visualization (improvement checkpoints only)...")

    num_checkpoints = len(adjacency_matrices)
    num_heads = adjacency_matrices[0].shape[0]
    n_nodes = adjacency_matrices[0].shape[1]
    tick_labels = _shorten_labels(feature_names)

    fig, axes = plt.subplots(num_checkpoints, num_heads, figsize=(5 * num_heads, 4 * num_checkpoints))
    fig.suptitle(
        f'What Drives {target_name} — Val-Improvement Checkpoints ({num_checkpoints})\n'
        f'Read the {target_name} row for driver weights (col j → row {target_name})',
        fontsize=16, fontweight='bold', y=1.01,
    )

    if num_checkpoints == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_heads == 1:
        axes = np.expand_dims(axes, axis=1)

    vmin, vmax = 0, 0.8
    mid = (vmin + vmax) * 0.5
    cell_fs = max(3.0, min(6.0, 80 / n_nodes))

    for epoch_idx, A_heads in enumerate(adjacency_matrices):
        ep_num = improved_epoch_nums[epoch_idx]
        for h in range(num_heads):
            ax = axes[epoch_idx, h]
            im = ax.imshow(A_heads[h], cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='equal')

            for i in range(n_nodes):
                for j in range(n_nodes):
                    value = A_heads[h, i, j]
                    text_color = 'white' if value > mid else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=text_color, fontsize=cell_fs, fontweight='bold')

            ax.set_xticks(range(n_nodes))
            ax.set_yticks(range(n_nodes))
            show_x = epoch_idx == num_checkpoints - 1
            show_y = h == 0
            ax.set_xticklabels(tick_labels if show_x else [], rotation=55, ha='right', fontsize=5)
            ax.set_yticklabels(tick_labels if show_y else [], fontsize=5)

            if epoch_idx == 0:
                ax.set_title(f'Head {h + 1}', fontsize=12, fontweight='bold')
            if h == 0:
                ax.set_ylabel(f'Epoch {ep_num}', fontsize=11, fontweight='bold')

            demand_row = A_heads[h, dataset.target_idx, :]
            drivers = demand_row.copy()
            drivers[dataset.target_idx] = 0
            ax.text(0.5, -0.12, f'driver mass={drivers.sum():.3f}',
                    transform=ax.transAxes, ha='center', fontsize=8, style='italic')

    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label('Adjacency Weight', fontsize=14)
    plt.tight_layout()
    plt.savefig(EVOLUTION_PNG, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {EVOLUTION_PNG}")
    plt.close()
else:
    print("📊 No validation improvements — skipping adjacency evolution plot.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
actual_epochs = len(train_losses)
axes[0].plot(range(1, actual_epochs + 1), train_losses, 'o-', label='Train Loss', linewidth=2, markersize=6)
axes[0].plot(range(1, actual_epochs + 1), val_losses, 's-', label='Val Loss', linewidth=2, markersize=6)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title(f'Training Curves - {actual_epochs} Epochs', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

if adjacency_matrices:
    driver_sums = []
    for A in adjacency_matrices:
        row = A[0, dataset.target_idx, :].copy()
        row[dataset.target_idx] = 0
        driver_sums.append(row.sum())
    axes[1].plot(improved_epoch_nums, driver_sums, 'o-', label=f'{target_name} driver mass', linewidth=2, markersize=6)
    axes[1].set_title(f'Driver Mass Into {target_name}', fontsize=14, fontweight='bold')
else:
    axes[1].text(0.5, 0.5, 'No checkpoints saved', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Driver Mass', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Sum of off-diag target row', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(STATS_PNG, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {STATS_PNG}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"\n📊 Final Results (Epoch {len(train_losses)}):")
print(f"   Train Loss: {train_losses[-1]:.6f}")
print(f"   Val Loss:   {val_losses[-1]:.6f}")
if adjacency_matrices:
    row = adjacency_matrices[-1][0, dataset.target_idx, :].copy()
    row[dataset.target_idx] = 0
    top = sorted(zip(feature_names, row), key=lambda x: -x[1])[:5]
    print(f"\n🎯 Top drivers of {target_name} (last checkpoint, epoch {improved_epoch_nums[-1]}):")
    for name, w in top:
        print(f"   {name}: {w:.4f}")
print(f"\n✅ Outputs in {ADJ_OUT_DIR}/, {EVOLUTION_PNG}, {STATS_PNG}")
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

with torch.no_grad():
    for X_batch, y_batch in DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        y_pred, _ = model(X_batch)
        test_loss += nn.MSELoss()(y_pred, y_batch).item() * X_batch.size(0)
        all_preds.append(y_pred.cpu().numpy())
        all_trues.append(y_batch.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_trues = np.concatenate(all_trues, axis=0)
test_loss /= len(test_subset)

mse = mean_squared_error(all_trues.flatten(), all_preds.flatten())
mae = mean_absolute_error(all_trues.flatten(), all_preds.flatten())
rmse = np.sqrt(mse)
r2 = r2_score(all_trues.flatten(), all_preds.flatten())

print(f"\n📈 Test Set Results:")
print(f"   Test Loss (MSE):  {test_loss:.6f}")
print(f"   Test RMSE:        {rmse:.6f}")
print(f"   Test MAE:         {mae:.6f}")
print(f"   Test R² Score:    {r2:.6f}")
print("\n✅ TESTING COMPLETE")
print("="*80)
