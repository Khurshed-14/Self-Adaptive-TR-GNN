from tqdm import tqdm
import torch
import torch.nn as nn
import os
import numpy as np
import time  # <--- Added import for timing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad(), tqdm(loader, desc="Validating", leave=False) as pbar:
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            pred, _ = model(X)      # <-- unpack tuple, discard A
            loss = criterion(pred, Y)
            total_loss += loss.item()
    return total_loss / len(loader)

# Training function with TensorBoard logging
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-4,
    device='cuda',
    patience=10,
    scheduler_patience=4,
    scheduler_factor=0.5,
    save_path="ISO_NE_Small_Dataset_Run2",
    writer=None,
    weight_decay=1e-5,
    lambda_smooth=0.01,     # <-- L2 temporal consistency weight
    lambda_sparse=1e-4,     # <-- L1 sparsity weight (optional, kept from before)
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-6
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_history = []
    val_history = []
    start_total_time = time.time()
    best_model_time = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for X, Y in loop:
            X, Y = X.to(device), Y.to(device)

            pred, A = model(X)          # <-- unpack (Y_hat, A)

            # Prediction loss
            pred_loss = criterion(pred, Y)
            
            # CRITICAL: Strong graph learning regularization
            # The graph MUST learn diverse structures or prediction will suffer
            off_diag_mask = 1 - torch.eye(A.size(1), device=A.device)
            A_off_diag = A * off_diag_mask.unsqueeze(0)
            
            # Variance-based regularization: force structure diversity
            A_mean = A_off_diag.mean()
            variance = ((A_off_diag - A_mean) ** 2).mean()
            
            # Min-max spread: encourage edges to span the full [0.2, 0.8] range
            # This forces learning of differentiated edge strengths
            edge_spread = (A_off_diag.max() - A_off_diag.min())
            
            # Regularization terms weighted EQUALLY with prediction loss
            # to create strong gradient pressure on graph parameters
            graph_loss = (-variance * 100.0) + (-edge_spread * 50.0)
            
            # CRITICAL: Use weighted combination with EQUAL or HIGHER regularization
            # If pred_loss ~0.17 and graph_loss contribution is ~-1, we need to scale
            loss = pred_loss + graph_loss * 0.2  # Adjust multiplier to balance forces

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_train_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion, device)

        train_history.append(avg_train_loss)
        val_history.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            best_model_time = time.time() - start_total_time
            print(f"✅ New best model saved (Val Loss: {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\n⛔ Early stopping triggered after {patience} epochs without improvement.")
            break

    total_duration = time.time() - start_total_time

    def format_duration(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{int(h)}h {int(m)}m {int(s)}s"
        return f"{int(m)}m {int(s)}s"

    print("\n" + "="*50)
    print("               TIMING REPORT               ")
    print("="*50)
    print(f"⏱️  Time to reach Best Model: {format_duration(best_model_time)}")
    print(f"⏱️  Total Training Duration:  {format_duration(total_duration)}")
    print("="*50 + "\n")

    print(f"Loading best model from {save_path} (Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(torch.load(save_path, map_location=device))

    plt.figure(figsize=(8, 5))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(save_path + "_learning_curve.png", dpi=200)
    plt.close()

    print("Training complete. TensorBoard logs saved.")
    return model


# Testing function with TensorBoard logging
def test_model(dataset, model, test_loader, device='cuda', writer=None):
    model.eval()
    preds_all, trues_all = [], []

    with torch.no_grad(), tqdm(test_loader, desc="Testing") as pbar:
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            out, _ = model(X)
            preds_all.append(out.cpu().numpy())
            trues_all.append(Y.cpu().numpy())

    preds_flat = np.concatenate(preds_all, axis=0).flatten()
    trues_flat = np.concatenate(trues_all, axis=0).flatten()

    assert preds_flat.shape == trues_flat.shape
    assert not np.any(np.isnan(preds_flat)), "Predictions contain NaN"
    assert not np.any(np.isnan(trues_flat)), "Targets contain NaN"
    
    mse_scaled = mean_squared_error(trues_flat, preds_flat)
    mae_scaled = mean_absolute_error(trues_flat, preds_flat)
    r2_scaled  = r2_score(trues_flat, preds_flat)
    
    n          = len(preds_flat)
    n_features = len(dataset.feature_names)
    target_col = dataset.target_idx   # 0 = demand

    dummy_preds = np.zeros((n, n_features))
    dummy_trues = np.zeros((n, n_features))
    dummy_preds[:, target_col] = preds_flat
    dummy_trues[:, target_col] = trues_flat

    preds_unscaled = dataset.scaler.inverse_transform(dummy_preds)[:, target_col]
    trues_unscaled = dataset.scaler.inverse_transform(dummy_trues)[:, target_col]

    mse = mean_squared_error(trues_unscaled, preds_unscaled)
    mae = mean_absolute_error(trues_unscaled, preds_unscaled)
    r2  = r2_score(trues_unscaled, preds_unscaled)

    print(f"\nTest Results ({n:,} points, unscaled MW):")
    print(f"Scaled Metrics:   MSE = {mse_scaled:.4f} | MAE = {mae_scaled:.4f} | R² = {r2_scaled:.4f}")
    print(f"Unscaled (MW):    MSE = {mse:.4f} | MAE = {mae:.4f} | R² = {r2:.4f}\n")
    
    if writer:
        writer.add_scalar('Test_Metrics/MSE', mse, 1)
        writer.add_scalar('Test_Metrics/MAE', mae, 1)
        writer.add_scalar('Test_Metrics/R2', r2,  1)
        print("Test metrics logged to TensorBoard.")

    return np.concatenate(preds_all, axis=0), np.concatenate(trues_all, axis=0)

def get_cluster_prior(dataset, n_clusters=5):
    """
    Computes a cluster mask based on feature correlations.
    Returns a tensor (N, N) where A_ij = 1 if i and j are in the same cluster.
    """
    # 1. Get raw data from the dataset
    # We use the unscaled data to capture true correlations
    df = dataset.df_numeric
    
    # 2. Compute Correlation Matrix (N x N)
    corr_matrix = df.corr().fillna(0).values
    
    # 3. Perform Clustering (e.g., K-Means on the correlation features)
    # This groups features that behave similarly
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(corr_matrix)
    
    # 4. Create the Prior Adjacency Matrix
    N = len(labels)
    prior_adj = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                prior_adj[i, j] = 1.0  # Same cluster connection
            else:
                prior_adj[i, j] = 0.0  # Different cluster (weak connection)
                
    # Normalize or scale if needed, but 0/1 is fine for a bias
    return torch.FloatTensor(prior_adj)


def test_model_stepwise(dataset, model, test_loader, device='cuda', writer=None, save_plot_path="stepwise_error.png"):
    model.eval()
    preds_all, trues_all = [], []

    with torch.no_grad(), tqdm(test_loader, desc="Testing") as pbar:
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            out, _ = model(X)
            preds_all.append(out.cpu().numpy())
            trues_all.append(Y.cpu().numpy())

    # Shape: (Total_Samples, T_out)
    # Note: These are the SCALED values straight from the model
    preds_concat = np.concatenate(preds_all, axis=0)
    trues_concat = np.concatenate(trues_all, axis=0)

    N_samples, T_out = preds_concat.shape
    n_features = len(dataset.feature_names)
    target_col = dataset.target_idx   # 0 = demand

    # --- A. Calculate SCALED Metrics ---
    mse_scaled_per_step = np.mean((preds_concat - trues_concat)**2, axis=0)
    mae_scaled_per_step = np.mean(np.abs(preds_concat - trues_concat), axis=0)
    
    overall_mse_scaled = mean_squared_error(trues_concat.flatten(), preds_concat.flatten())
    overall_mae_scaled = mean_absolute_error(trues_concat.flatten(), preds_concat.flatten())

    # --- B. Calculate UNSCALED Metrics ---
    preds_flat = preds_concat.flatten()
    trues_flat = trues_concat.flatten()

    dummy_preds = np.zeros((N_samples * T_out, n_features))
    dummy_trues = np.zeros((N_samples * T_out, n_features))
    
    dummy_preds[:, target_col] = preds_flat
    dummy_trues[:, target_col] = trues_flat

    preds_unscaled_flat = dataset.scaler.inverse_transform(dummy_preds)[:, target_col]
    trues_unscaled_flat = dataset.scaler.inverse_transform(dummy_trues)[:, target_col]

    preds_unscaled = preds_unscaled_flat.reshape(N_samples, T_out)
    trues_unscaled = trues_unscaled_flat.reshape(N_samples, T_out)

    mse_per_step = np.mean((preds_unscaled - trues_unscaled)**2, axis=0)
    mae_per_step = np.mean(np.abs(preds_unscaled - trues_unscaled), axis=0)

    overall_mse = mean_squared_error(trues_unscaled_flat, preds_unscaled_flat)
    overall_mae = mean_absolute_error(trues_unscaled_flat, preds_unscaled_flat)

    # --- C. Print Results ---
    print(f"\n--- Overall Test Results ({N_samples:,} points) ---")
    print(f"[Scaled]   Overall MSE = {overall_mse_scaled:.4f} | Overall MAE = {overall_mae_scaled:.4f}")
    print(f"[Unscaled] Overall MSE = {overall_mse:.4f} | Overall MAE = {overall_mae:.4f} (MW)")

    steps_to_report = [60, 120, 180, 240]  
    print("\n--- Error Decomposition at Specific Steps ---")
    for step in steps_to_report:
        idx = step - 1
        print(f"Step {step:3d}:")
        print(f"  Scaled   -> MSE = {mse_scaled_per_step[idx]:.4f} | MAE = {mae_scaled_per_step[idx]:.4f}")
        print(f"  Unscaled -> MSE = {mse_per_step[idx]:.4f} | MAE = {mae_per_step[idx]:.4f}")

    # --- D. Plot the "Prediction Step vs. Error" curves (Side-by-Side) ---
    steps_axis = np.arange(1, T_out + 1)
    
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5))

    color_mse = 'tab:red'
    color_mae = 'tab:blue'

    # 1. Unscaled Plot (Left)
    ax1.set_xlabel('Prediction Step (Horizon)')
    ax1.set_ylabel('MSE (Unscaled)', color=color_mse)
    ax1.plot(steps_axis, mse_per_step, color=color_mse, label='MSE')
    ax1.tick_params(axis='y', labelcolor=color_mse)

    ax2 = ax1.twinx()  
    ax2.set_ylabel('MAE (Unscaled)', color=color_mae)  
    ax2.plot(steps_axis, mae_per_step, color=color_mae, label='MAE', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_mae)
    ax1.set_title('Unscaled Error Evolution')

    # 2. Scaled Plot (Right)
    ax3.set_xlabel('Prediction Step (Horizon)')
    ax3.set_ylabel('MSE (Scaled)', color=color_mse)
    ax3.plot(steps_axis, mse_scaled_per_step, color=color_mse, label='MSE')
    ax3.tick_params(axis='y', labelcolor=color_mse)

    ax4 = ax3.twinx()
    ax4.set_ylabel('MAE (Scaled)', color=color_mae)
    ax4.plot(steps_axis, mae_scaled_per_step, color=color_mae, label='MAE', linestyle='--')
    ax4.tick_params(axis='y', labelcolor=color_mae)
    ax3.set_title('Scaled Error Evolution')

    # Highlight the specific steps on both plots
    for step in steps_to_report:
        ax1.axvline(x=step, color='gray', linestyle=':', alpha=0.6)
        ax3.axvline(x=step, color='gray', linestyle=':', alpha=0.6)

    fig.tight_layout()  
    plt.savefig(save_plot_path, dpi=300)
    plt.show()
    print(f"\nPlots saved to {save_plot_path}")

    return preds_concat, trues_concat, mse_per_step, mae_per_step