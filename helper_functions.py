from tqdm import tqdm
import torch
import torch.nn as nn
import os
import numpy as np
import time  # <--- Added import for timing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


def validate(model, loader, criterion, device, return_adjacency=False):
    model.eval()
    total_loss = 0.0
    epoch_A = None
    with torch.no_grad(), tqdm(loader, desc="Validating", leave=False) as pbar:
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            pred, A = model(X)
            loss = criterion(pred, Y)
            total_loss += loss.item()
            if return_adjacency and epoch_A is None:
                epoch_A = A.detach().cpu().numpy()
    avg_loss = total_loss / len(loader)
    if return_adjacency:
        return avg_loss, epoch_A
    return avg_loss


def _adjacency_save_path(save_path):
    base, _ = os.path.splitext(save_path)
    return f"{base}_adjacency.npy"


def _adjacency_plot_path(save_path):
    base, _ = os.path.splitext(save_path)
    return f"{base}_adjacency.png"


def _shorten_labels(names, max_len=14):
    out = []
    for name in names:
        s = str(name)
        out.append(s if len(s) <= max_len else s[: max_len - 1] + "\u2026")
    return out


def _normalize_adjacency(A):
    """Ensure shape (num_heads, N, N) for plotting."""
    A = np.asarray(A)
    if A.ndim == 2:
        return A[np.newaxis, ...]
    return A


def _save_adjacency_plot(
    A,
    plot_path,
    feature_names=None,
    demand_targeting=False,
    target_idx=None,
    val_loss=None,
    vmin=0.0,
    vmax=0.8,
):
    """Save a labeled adjacency heatmap, overwriting plot_path on each call."""
    A_heads = _normalize_adjacency(A)
    num_heads, n, _ = A_heads.shape

    if feature_names is not None and len(feature_names) == n:
        labels = _shorten_labels(feature_names)
    else:
        labels = [f"F{i}" for i in range(n)]

    plot_dir = os.path.dirname(plot_path)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    side = max(6.0, n * 0.35)
    fig, axes = plt.subplots(1, num_heads, figsize=(side * num_heads + 0.8, side + 0.6))
    if num_heads == 1:
        axes = [axes]

    mid = (vmin + vmax) * 0.5
    cell_fs = max(3.5, min(7.0, 90 / n))
    last_im = None
    for h, ax in enumerate(axes):
        mat = A_heads[h]
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

    if demand_targeting and feature_names is not None and target_idx is not None:
        target_name = feature_names[target_idx]
        title = f'What Drives {target_name} — Best Model'
    else:
        title = 'Adjacency Matrix — Best Model'
    if val_loss is not None:
        title += f'\nVal Loss: {val_loss:.6f}'

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.colorbar(last_im, ax=axes, label='Adjacency Weight', shrink=0.8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

def _adjacency_l2_reg(A, demand_targeting=False, target_idx=None):
    """
    L2 penalty on off-diagonal adjacency weights.

    When demand_targeting is True, the target row is excluded so regularization
    does not penalize learned drivers into the demand node (target_in graphs).
    """
    N = A.size(-1)
    identity_mask = torch.eye(N, device=A.device)
    off_diag_mask = 1.0 - identity_mask
    if demand_targeting:
        if target_idx is None:
            raise ValueError("target_idx is required when demand_targeting_adjacency=True")
        off_diag_mask[target_idx, :] = 0.0
    off_diagonal_A = A * off_diag_mask
    return torch.mean(off_diagonal_A ** 2)


def _build_optimizer(model, lr, weight_decay):
    """Graph-learning params: no weight decay; all other params: normal decay."""
    if hasattr(model, 'graph_learn'):
        graph_param_ids = {id(p) for p in model.graph_learn.parameters()}
        param_groups = [
            {
                'params': [p for p in model.parameters() if id(p) in graph_param_ids],
                'lr': lr,
                'weight_decay': 0.0,
            },
            {
                'params': [p for p in model.parameters() if id(p) not in graph_param_ids],
                'lr': lr,
                'weight_decay': weight_decay,
            },
        ]
        return torch.optim.Adam(param_groups)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


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
    adjacency_save_path=None,
    adjacency_plot_path=None,
    feature_names=None,
    writer=None,
    weight_decay=1e-5,
    l2_alpha=0.005,
    demand_targeting_adjacency=False,
    target_idx=None,
    return_stats=False,
):
    model = model.to(device)
    optimizer = _build_optimizer(model, lr, weight_decay)
    criterion = nn.MSELoss()

    if demand_targeting_adjacency and target_idx is None and getattr(model, 'target_idx', None) is not None:
        target_idx = model.target_idx

    if adjacency_save_path is None:
        adjacency_save_path = _adjacency_save_path(save_path)
    if adjacency_plot_path is None:
        adjacency_plot_path = _adjacency_plot_path(save_path)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-7
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    adj_dir = os.path.dirname(adjacency_save_path)
    if adj_dir and not os.path.exists(adj_dir):
        os.makedirs(adj_dir, exist_ok=True)
    plot_dir = os.path.dirname(adjacency_plot_path)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    train_history = []
    val_history = []
    start_total_time = time.time()
    best_model_time = 0
    best_epoch = 0
    epochs_run = 0
    epoch_times = []
    peak_gpu_train_mb = 0.0
    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        epochs_run = epoch
        model.train()
        total_mse = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for X, Y in loop:
            X, Y = X.to(device), Y.to(device)

            pred, A = model(X)
            mse_loss = criterion(pred, Y)
            l2_reg = _adjacency_l2_reg(
                A,
                demand_targeting=demand_targeting_adjacency,
                target_idx=target_idx,
            )
            loss = mse_loss + (l2_alpha * l2_reg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_mse += mse_loss.item()
            loop.set_postfix(mse=f"{mse_loss.item():.4f}")

        avg_train_loss = total_mse / len(train_loader)
        val_loss, epoch_A = validate(
            model, val_loader, criterion, device, return_adjacency=True
        )

        train_history.append(avg_train_loss)
        val_history.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if writer is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            if epoch_A is not None:
                np.save(adjacency_save_path, epoch_A)
                _save_adjacency_plot(
                    epoch_A,
                    adjacency_plot_path,
                    feature_names=feature_names,
                    demand_targeting=demand_targeting_adjacency,
                    target_idx=target_idx,
                    val_loss=best_val_loss,
                )
            best_model_time = time.time() - start_total_time
            print(
                f"✅ New best model saved (Val Loss: {best_val_loss:.6f}) "
                f"-> {save_path}"
            )
            if epoch_A is not None:
                print(f"   Adjacency matrix updated -> {adjacency_save_path}")
                print(f"   Adjacency plot updated   -> {adjacency_plot_path}")
        else:
            epochs_no_improve += 1
            print(f"⚠️  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\n⛔ Early stopping triggered after {patience} epochs without improvement.")
            break

        epoch_times.append(time.perf_counter() - epoch_start)
        if use_cuda:
            peak_gpu_train_mb = max(
                peak_gpu_train_mb,
                torch.cuda.max_memory_allocated() / (1024 ** 2),
            )

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

    train_stats = {
        "total_time_s": total_duration,
        "time_to_best_s": best_model_time,
        "best_epoch": best_epoch,
        "total_epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
        "avg_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "gpu_mem_training_mb": peak_gpu_train_mb,
    }
    if return_stats:
        return model, train_stats
    return model


# Testing function with TensorBoard logging
def test_model(dataset, model, test_loader, device='cuda', writer=None, return_metrics=False):
    model.eval()
    preds_all, trues_all = [], []
    batch_times = []
    peak_gpu_infer_mb = 0.0
    use_cuda = torch.cuda.is_available() and device == "cuda"
    if return_metrics and use_cuda:
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad(), tqdm(test_loader, desc="Testing") as pbar:
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            if return_metrics:
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out, _ = model(X)
                if use_cuda:
                    torch.cuda.synchronize()
                batch_times.append(time.perf_counter() - t0)
                if use_cuda:
                    peak_gpu_infer_mb = max(
                        peak_gpu_infer_mb,
                        torch.cuda.max_memory_allocated() / (1024 ** 2),
                    )
            else:
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

    preds = np.concatenate(preds_all, axis=0)
    trues = np.concatenate(trues_all, axis=0)
    if return_metrics:
        test_metrics = {
            "test_mse": float(mse_scaled),
            "test_mae": float(mae_scaled),
            "test_r2": float(r2_scaled),
            "gpu_mem_inference_mb": peak_gpu_infer_mb,
            "avg_inference_batch_ms": float(np.mean(batch_times)) * 1000 if batch_times else 0.0,
        }
        return preds, trues, test_metrics
    return preds, trues

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


def test_model_stepwise(
    dataset,
    model,
    test_loader,
    device='cuda',
    writer=None,
    output_dir=".",
    save_plot_name="stepwise_error.png",
    metrics_csv_name="stepwise_metrics.csv",
):
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

    for ax in (ax1, ax3):
        ax.set_xlim(1, T_out)
        ax.set_xticks([1, 60, 120, 180, 240])
        ax.set_xticklabels(['1', '60', '120', '180', '240'])

    # Highlight the specific steps on both plots
    for step in steps_to_report:
        ax1.axvline(x=step, color='gray', linestyle=':', alpha=0.6)
        ax3.axvline(x=step, color='gray', linestyle=':', alpha=0.6)

    fig.tight_layout()  
    os.makedirs(output_dir, exist_ok=True)
    save_plot_path = os.path.join(output_dir, save_plot_name)
    metrics_csv_path = os.path.join(output_dir, metrics_csv_name)

    metrics_df = pd.DataFrame({
        'step': steps_axis,
        'mse_scaled': mse_scaled_per_step,
        'mae_scaled': mae_scaled_per_step,
        'mse_unscaled': mse_per_step,
        'mae_unscaled': mae_per_step,
    })
    summary_df = pd.DataFrame([
        {
            'step': 'overall',
            'mse_scaled': overall_mse_scaled,
            'mae_scaled': overall_mae_scaled,
            'mse_unscaled': overall_mse,
            'mae_unscaled': overall_mae,
        }
    ])
    pd.concat([metrics_df, summary_df], ignore_index=True).to_csv(metrics_csv_path, index=False)

    plt.savefig(save_plot_path, dpi=300)
    plt.show()
    print(f"\nPlots saved to {save_plot_path}")
    print(f"Metrics saved to {metrics_csv_path}")

    return preds_concat, trues_concat, mse_per_step, mae_per_step