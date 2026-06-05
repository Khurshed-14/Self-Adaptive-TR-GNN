import numpy as np
from helper_functions import _save_adjacency_plot
from dataset_classes import ISO_NE

# Load the dataset for the appropriate feature names (if needed for labeling the adjacency plot)
dataset = ISO_NE(
        csv_path=r"data\iso_ne\selected_data_ISONE.csv",
        T_in=72,
        T_out=240,
        lag_hours=[1, 12, 24, 168],
        rolling_windows=[12, 24],
    )

# Replace this with the actual path to your saved adjacency .npy file
adj_matrix = np.load(r"notebooks\experiments\ISO_NE\adjacency\files\Sens_Vary_GCN_Layer_to_2_GCN2_Hidden64_Kernel7_Dil3_best_model_adjacency.npy")

_save_adjacency_plot(
    A=adj_matrix, 
    plot_path="my_adjacency_plot.pdf",
    feature_names=dataset.feature_names, # Replace with your dataset.feature_names for labeled axes
    vmin=0.0, 
    vmax=1.0  # Adjust depending on your graph weights
)

print("Plot saved to my_adjacency_plot.png")