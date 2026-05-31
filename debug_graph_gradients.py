"""
Debug script to check if graph learning parameters receive gradients
and verify gradient flow through the model.
"""
import torch
import torch.nn as nn
from dataset_classes import ISO_NE
from models_with_temporal_graph import TR_GNN_MultiScale
from torch.utils.data import DataLoader, Subset
import numpy as np

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load dataset
print("\n📂 Loading dataset...")
dataset = ISO_NE(
    csv_path="data/iso_ne/selected_data_ISONE.csv",
    T_in=72,
    T_out=240,
    lag_hours=[1, 12, 24, 168],
    rolling_windows=[12, 24],
)

# Apply scaler and get subset
from sklearn.preprocessing import StandardScaler
total_len = len(dataset.df_numeric)
train_split_idx = int(0.6 * total_len)

scaler = StandardScaler()
scaler.fit(dataset.df_numeric.iloc[:train_split_idx].values)
dataset.data = torch.tensor(scaler.transform(dataset.df_numeric.values), dtype=torch.float32)

# Use subset for debugging
subset_size = 5000
subset_indices = list(range(0, train_split_idx))[:subset_size]
subset = Subset(dataset, subset_indices)
train_loader = DataLoader(subset, batch_size=32, shuffle=False)

# Initialize model
print("\n🏗️  Initializing model...")
model = TR_GNN_MultiScale(
    N=19, T_in=72, T_out=240, d=32, hidden_dim=64,
    dropout_temporal=0.2, dropout_gcn=0.2, dropout_forecast=0.3,
    GCN_Layer=5, kernel_size=7, dilation=3, graph_temperature=3.0
).to(device)

# Get one batch
X, Y = next(iter(train_loader))
X, Y = X.to(device), Y.to(device)

print(f"\nBatch shape: X={X.shape}, Y={Y.shape}")

# Get graph learning module
graph_module = None
for name, module in model.named_modules():
    if module.__class__.__name__ == 'TemporalGraphLearning':
        graph_module = module
        print(f"Found TemporalGraphLearning: {name}")
        break

if graph_module is None:
    print("❌ TemporalGraphLearning not found!")
    exit()

# Check parameters
print(f"\n📊 Graph Module Parameters:")
for name, param in graph_module.named_parameters():
    if param.requires_grad:
        print(f"   {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        print(f"      Initial values (first 5): {param.data.flatten()[:5]}")

# Forward pass
print(f"\n🚀 Forward pass...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# STEP 1: Forward pass
pred, A = model(X)
print(f"   Pred shape: {pred.shape}")
print(f"   Adjacency A shape: {A.shape}")
print(f"   Adjacency values (first batch, first 5x5):")
print(A[0, :5, :5].detach().cpu().numpy())

# STEP 2: Compute loss with regularization
pred_loss = criterion(pred, Y)

# Add regularization
off_diag_mask = 1 - torch.eye(A.size(1), device=A.device)
A_off_diag = A * off_diag_mask.unsqueeze(0)
A_mean = A_off_diag.mean()
variance = ((A_off_diag - A_mean) ** 2).mean()
edge_spread = (A_off_diag.max() - A_off_diag.min())
graph_loss = (-variance * 100.0) + (-edge_spread * 50.0)
total_loss = pred_loss + graph_loss * 0.2

print(f"\n💔 Loss Computation:")
print(f"   Pred loss: {pred_loss.item():.6f}")
print(f"   Variance: {variance.item():.6f}")
print(f"   Edge spread: {edge_spread.item():.6f}")
print(f"   Graph loss: {graph_loss.item():.6f}")
print(f"   Total loss: {total_loss.item():.6f}")

# STEP 3: Backward pass
print(f"\n📉 Backward pass...")
optimizer.zero_grad()
total_loss.backward()

# STEP 4: Check gradients
print(f"\n🔍 Gradient Analysis:")
print(f"   Graph module parameter gradients:")
for name, param in graph_module.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        max_grad = param.grad.abs().max().item()
        min_grad = param.grad.abs().min().item()
        print(f"   {name}:")
        print(f"      Grad norm: {grad_norm:.6e}, Max: {max_grad:.6e}, Min: {min_grad:.6e}")
    else:
        print(f"   {name}: ❌ NO GRADIENT!")

# Check if A has gradients
print(f"\n🔍 Adjacency Gradient:")
if A.requires_grad:
    print(f"   A.requires_grad = True ✅")
    if A.grad is not None:
        print(f"   A.grad exists: norm={A.grad.norm().item():.6e}")
    else:
        print(f"   ❌ A has no grad even though requires_grad=True")
else:
    print(f"   ❌ A.requires_grad = False (PROBLEM!)")

# STEP 5: Optimizer step
print(f"\n⚙️  Optimizer step...")
optimizer.step()

# STEP 6: Check if parameters changed
print(f"\n📊 Parameter Changes:")
for name, param in graph_module.named_parameters():
    if param.grad is not None:
        print(f"   {name}: Mean change = {param.grad.mean().abs().item():.6e}")

# STEP 7: Forward pass again to see if adjacency changed
pred2, A2 = model(X)
print(f"\n🔄 Adjacency After Optimizer Step (first batch, first 5x5):")
print(A2[0, :5, :5].detach().cpu().numpy())

A_changed = not np.allclose(A[0].detach().cpu().numpy(), A2[0].detach().cpu().numpy(), atol=1e-6)
print(f"\n✅ Adjacency changed: {A_changed}")

if not A_changed:
    print("❌ PROBLEM: Adjacency not changing despite gradients!")
