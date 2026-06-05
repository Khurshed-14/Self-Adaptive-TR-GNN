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

# Graph / training config (aligned with full_training_graph_viz*.py)
GRAPH_STRUCTURE = 'full'       # 'full' or 'target_in' (demand-driver graph)
L2_ALPHA = 0.0005
GRAPH_TEMPERATURE = 1.0
SAVE_DIR = "Sens_Runs/ISO_NE"
RESULTS_CSV = os.path.join(SAVE_DIR, "ISO_NE_sensitivity_results.csv")

RESULT_COLUMNS = [
    "run_name",
    "param_vary",
    "vary_value",
    "is_base",
    "GCN_Layer",
    "hidden_dim",
    "kernel_size",
    "dilation",
    "test_mae",
    "test_mse",
    "test_r2",
    "total_time_s",
    "time_to_best_s",
    "best_epoch",
    "total_epochs_run",
    "gpu_mem_training_mb",
    "gpu_mem_inference_mb",
    "peak_gpu_mem_mb",
    "avg_epoch_time_s",
    "best_val_loss",
    "model_path",
]

# Sensitivity grids — vary one parameter at a time; base run only once
SENSITIVITY_GRIDS = {
    "GCN_Layer": [1, 2, 3, 5, 7],
    "dilation": [1, 2, 3, 5],
    "kernel_size": [3, 5, 7, 11],
    "hidden_dim": [32, 64, 128, 256],
}


def build_sensitivity_configs(base_hparams):
    """
    One-at-a-time sweeps: for each grid, override a single key while others stay at base.
    The full base configuration is scheduled exactly once (first time it appears).
    """
    configs_to_run = []
    base_added = False

    for param_to_vary, values in SENSITIVITY_GRIDS.items():
        for val in values:
            current = base_hparams.copy()
            current[param_to_vary] = val

            is_base = all(
                current[k] == base_hparams[k] for k in SENSITIVITY_GRIDS.keys()
            )
            if is_base:
                if base_added:
                    continue
                base_added = True
                run_name = "Sens_BASE_GCN5_Hidden64_Kernel7_Dil3"
            else:
                run_name = (
                    f"Sens_Vary_{param_to_vary}_to_{val}_"
                    f"GCN{current['GCN_Layer']}_Hidden{current['hidden_dim']}_"
                    f"Kernel{current['kernel_size']}_Dil{current['dilation']}"
                )

            configs_to_run.append((run_name, current, param_to_vary, val, is_base))

    return configs_to_run


def prepare_data_loaders(dataset, batch_size=32):
    total_len = len(dataset.df_numeric)
    train_split_idx = int(0.6 * total_len)
    val_split_idx = int(0.8 * total_len)

    print(f"Raw data length: {total_len}")
    print(f"Scaler 'train_size' (raw rows): {train_split_idx}")
    print(f"Scaler 'val_end' (raw rows): {val_split_idx}")

    scaler = StandardScaler()
    scaler.fit(dataset.df_numeric.iloc[:train_split_idx].values.astype(np.float32))

    print("\n🔍 Applying scaler (train split only)...")
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
    print(
        f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, "
        f"Test samples: {len(test_idx)}"
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(
        f"\n🚚 DataLoaders ready. Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )
    return train_loader, val_loader, test_loader


def append_results_row(row, csv_path=RESULTS_CSV):
    """Append one sensitivity run to the master CSV (writes header if new)."""
    df_row = pd.DataFrame([row], columns=RESULT_COLUMNS)
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)
    print(f"💾 Results saved → {csv_path}")


def _ISO_NE_target_scale_std():
    """Standard deviation used for the load target (same fit as sensitivity training)."""
    dataset = ISO_NE(
        csv_path=r"data\iso_ne\selected_data_ISONE.csv",
        T_in=72,
        T_out=240,
        lag_hours=[1, 12, 24, 168],
        rolling_windows=[12, 24],
    )
    train_split_idx = int(0.6 * len(dataset.df_numeric))
    scaler = StandardScaler()
    scaler.fit(dataset.df_numeric.iloc[:train_split_idx].values.astype(np.float32))
    return float(scaler.scale_[dataset.target_idx])


def rescale_results_csv(csv_path=RESULTS_CSV, backup=True):
    """
    Convert unscaled test_mae / test_mse in an existing results CSV to scaled units
    using the training StandardScaler (no model re-run required). R² is unchanged.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty or "test_mae" not in df.columns or "test_mse" not in df.columns:
        raise ValueError(f"CSV missing expected columns: {csv_path}")

    # Skip if already scaled (typical scaled MSE is O(0.1–0.3), unscaled is O(1e6))
    if df["test_mse"].max() < 100:
        print(f"CSV already appears scaled — no changes: {csv_path}")
        return df

    target_std = _ISO_NE_target_scale_std()
    if backup:
        backup_path = csv_path + ".unscaled.bak"
        df.to_csv(backup_path, index=False)
        print(f"📋 Backup (unscaled) → {backup_path}")

    df["test_mae"] = df["test_mae"] / target_std
    df["test_mse"] = df["test_mse"] / (target_std ** 2)
    df.to_csv(csv_path, index=False)
    print(
        f"✅ Rescaled test_mae / test_mse (target_std={target_std:.4f}) → {csv_path}"
    )
    return df


def run_single_config(
    run_name,
    hparams,
    dataset,
    train_loader,
    val_loader,
    test_loader,
    device,
    demand_targeting,
    param_vary,
    vary_value,
    is_base,
):
    model_kwargs = dict(
        N=hparams["N"],
        T_in=hparams["T_in"],
        T_out=hparams["T_out"],
        d=hparams["d"],
        hidden_dim=hparams["hidden_dim"],
        GCN_Layer=hparams["GCN_Layer"],
        dropout_gcn=hparams["dropout_gcn"],
        dropout_temporal=hparams["dropout_temporal"],
        dropout_forecast=hparams["dropout_forecast"],
        kernel_size=hparams["kernel_size"],
        dilation=hparams["dilation"],
        graph_temperature=hparams["graph_temperature"],
    )
    if demand_targeting:
        model_kwargs["target_idx"] = dataset.target_idx
        model_kwargs["graph_structure"] = GRAPH_STRUCTURE

    model = TR_GNN_MultiScale(**model_kwargs).to(device)
    save_path = os.path.join(SAVE_DIR, f"{run_name}_best_model.pth")

    graph_tag = f"_{GRAPH_STRUCTURE}" if demand_targeting else ""
    log_dir = f"TR_GNN_ISO_NE/{run_name}{graph_tag}"
    writer = SummaryWriter(log_dir)
    writer.add_text("hparams", json.dumps(hparams, indent=2))

    print(f"\n🚀 Training: {run_name}")
    model, train_stats = train_model(
        model,
        train_loader,
        val_loader,
        epochs=hparams["epochs"],
        lr=hparams["lr"],
        device=device,
        scheduler_patience=hparams["scheduler_patience"],
        writer=writer,
        weight_decay=hparams["weight_decay"],
        l2_alpha=hparams["l2_alpha"],
        demand_targeting_adjacency=hparams["demand_targeting_adjacency"],
        target_idx=dataset.target_idx if demand_targeting else None,
        save_path=save_path,
        feature_names=dataset.feature_names,
        return_stats=True,
    )

    print(f"\n🧪 Testing: {run_name}\n")
    _, _, test_metrics = test_model(
        dataset=dataset,
        model=model,
        test_loader=test_loader,
        device=device,
        writer=writer,
        return_metrics=True,
    )
    writer.close()

    peak_gpu = max(
        train_stats["gpu_mem_training_mb"],
        test_metrics["gpu_mem_inference_mb"],
    )
    row = {
        "run_name": run_name,
        "param_vary": param_vary if not is_base else "baseline",
        "vary_value": vary_value if not is_base else "base",
        "is_base": is_base,
        "GCN_Layer": hparams["GCN_Layer"],
        "hidden_dim": hparams["hidden_dim"],
        "kernel_size": hparams["kernel_size"],
        "dilation": hparams["dilation"],
        "test_mae": test_metrics["test_mae"],
        "test_mse": test_metrics["test_mse"],
        "test_r2": test_metrics["test_r2"],
        "total_time_s": train_stats["total_time_s"],
        "time_to_best_s": train_stats["time_to_best_s"],
        "best_epoch": train_stats["best_epoch"],
        "total_epochs_run": train_stats["total_epochs_run"],
        "gpu_mem_training_mb": train_stats["gpu_mem_training_mb"],
        "gpu_mem_inference_mb": test_metrics["gpu_mem_inference_mb"],
        "peak_gpu_mem_mb": peak_gpu,
        "avg_epoch_time_s": train_stats["avg_epoch_time_s"],
        "best_val_loss": train_stats["best_val_loss"],
        "model_path": save_path,
    }

    append_results_row(row, csv_path=RESULTS_CSV)

    print(
        f"  → Test MAE (scaled) {row['test_mae']:.4f} | MSE (scaled) {row['test_mse']:.4f} | "
        f"R² (scaled) {row['test_r2']:.4f} | "
        f"Total {row['total_time_s']:.1f}s | To best {row['time_to_best_s']:.1f}s | "
        f"Epoch {row['best_epoch']}/{row['total_epochs_run']} | "
        f"GPU train {row['gpu_mem_training_mb']:.1f}MB infer {row['gpu_mem_inference_mb']:.1f}MB "
        f"peak {row['peak_gpu_mem_mb']:.1f}MB | {row['avg_epoch_time_s']:.2f}s/epoch"
    )
    return row


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demand_targeting = GRAPH_STRUCTURE == 'target_in'

    dataset = ISO_NE(
        csv_path=r"data\iso_ne\selected_data_ISONE.csv",
        T_in=72,
        T_out=240,
        lag_hours=[1, 12, 24, 168],
        rolling_windows=[12, 24],
    )

    if demand_targeting:
        target_name = dataset.feature_names[dataset.target_idx]
        print(
            f"Target: {target_name} (index {dataset.target_idx}), "
            f"graph_structure={GRAPH_STRUCTURE}"
        )

    train_loader, val_loader, test_loader = prepare_data_loaders(dataset)

    base_hparams = {
        "N": dataset.N,
        "T_in": 72,
        "T_out": 240,
        "d": 32,
        "hidden_dim": 64,
        "GCN_Layer": 5,
        "dropout_forecast": 0.3,
        "dropout_gcn": 0.2,
        "dropout_temporal": 0.2,
        "graph_temperature": GRAPH_TEMPERATURE,
        "graph_structure": GRAPH_STRUCTURE,
        "l2_alpha": L2_ALPHA,
        "lr": 1e-3,
        "scheduler_patience": 3,
        "batch_size": 32,
        "epochs": 100,
        "weight_decay": 1e-4,
        "kernel_size": 7,
        "dilation": 3,
        "demand_targeting_adjacency": demand_targeting,
    }

    configs_to_run = build_sensitivity_configs(base_hparams)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n⚙️ Sensitivity analysis — ISO_NE dataset")
    print(f"   Base: GCN=5, Hidden=64, Kernel=7, Dilation=3")
    print(f"   Total unique runs: {len(configs_to_run)}")
    print(f"   Results CSV: {RESULTS_CSV}")

    for run_idx, (run_name, hparams, param_vary, val, is_base) in enumerate(
        configs_to_run, 1
    ):
        tag = "BASE" if is_base else f"vary {param_vary}={val}"
        print(f"\n{'=' * 60}")
        print(f"[{run_idx}/{len(configs_to_run)}] {run_name}  ({tag})")
        print(f"{'=' * 60}")

        run_single_config(
            run_name,
            hparams,
            dataset,
            train_loader,
            val_loader,
            test_loader,
            device,
            demand_targeting,
            param_vary,
            val,
            is_base,
        )

    print(f"\n✅ ISO_NE sensitivity analysis completed. Results: {RESULTS_CSV}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--rescale-csv":
        rescale_results_csv()
    else:
        main()
