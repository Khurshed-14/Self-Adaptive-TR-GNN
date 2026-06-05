import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from dataset_classes import ISO_NE, SH_Dataset, AT
from helper_functions import test_model_stepwise
from models_with_temporal_graph import TR_GNN_MultiScale


GRAPH_STRUCTURE = "full"
MODEL_PATH = r"Self Adaptive\AT\models\kernel_size_11.pth"
SAVE_PLOT_PATH = r"Self Adaptive\AT\stepwise\kernel_size_11.pdf"
SAVE_METRICS_CSV_PATH = r"Self Adaptive\AT\stepwise\kernel_size_11.csv"


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


def build_model(dataset, device, hparams):
    demand_targeting = GRAPH_STRUCTURE == "target_in"
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

    return TR_GNN_MultiScale(**model_kwargs).to(device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained TR-GNN checkpoint and run stepwise test evaluation."
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to a saved .pth checkpoint",
    )
    parser.add_argument(
        "--save-plot-path",
        default=SAVE_PLOT_PATH,
        help="Path to save the stepwise error plot",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, e.g. cpu or cuda. Defaults to cuda if available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AT(
        csv_path=r"data\at\AT Dataset.csv",
        T_in=72,
        T_out=240,
        lag_hours=[1, 12, 24, 168],
        rolling_windows=[12, 24],
    )

    train_loader, val_loader, test_loader = prepare_data_loaders(dataset)

    hparams = {
        "N": dataset.N,
        "T_in": 72,
        "T_out": 240,
        "d": 32,
        "hidden_dim": 64,
        "GCN_Layer": 5,
        "dropout_forecast": 0.3,
        "dropout_gcn": 0.2,
        "dropout_temporal": 0.2,
        "graph_temperature": 1.0,
        "kernel_size": 11,
        "dilation": 3,
    }

    model = build_model(dataset, device, hparams)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    print(f"\n📦 Loading checkpoint: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    print("\n🧪 Running stepwise test evaluation...\n")
    preds, trues, mse_per_step, mae_per_step = test_model_stepwise(
        dataset=dataset,
        model=model,
        test_loader=test_loader,
        device=device,
        save_plot_name=args.save_plot_path,
        metrics_csv_name=SAVE_METRICS_CSV_PATH,
    )

    print("\n✅ Stepwise evaluation complete.")
    print(f"Predictions shape: {preds.shape}")
    print(f"Targets shape: {trues.shape}")
    print(f"Saved plot: {args.save_plot_path}")
    print(f"First 5 step MAE values: {mae_per_step[:5]}")
    print(f"First 5 step MSE values: {mse_per_step[:5]}")


if __name__ == "__main__":
    main()