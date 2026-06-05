import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_learning_curves(file_map, output_filename="learning_curves.pdf", title=None):
    """
    Plots train and validation loss from multiple CSV files.
    
    Parameters:
    - file_map: Dictionary where keys are experiment names (e.g., 'dilation_1') 
                and values are file paths (e.g., 'dilation_1.csv').
    - output_filename: Name of the PDF file to save.
    """
    
    # 1. Load and aggregate the data
    dfs = []
    for name, file in file_map.items():
        df = pd.read_csv(file)
        df['experiment'] = name
        dfs.append(df)
        
    combined_df = pd.concat(dfs)

    # 2. Reshape the data for seaborn (tidy format)
    # This transforms the dataframe to have columns: Step, experiment, Loss Type, Loss Value
    df_melted = combined_df.melt(
        id_vars=['Step', 'experiment'], 
        value_vars=['train_loss', 'val_loss'], 
        var_name='Loss Type', 
        value_name='Loss Value'
    )

    # 3. Create the plot
    sns.set_theme(style="whitegrid")
    
    # Use FacetGrid to separate Train and Val loss into two charts
    g = sns.FacetGrid(df_melted, col="Loss Type", hue="experiment", 
                      height=5, aspect=1.5, sharey=False)
    g.map(sns.lineplot, "Step", "Loss Value")
    
    # Add styling and legends
    g.add_legend(title="Experiment")
    g.set_titles("{col_name}")
    g.set_axis_labels("Step", "Loss")

    # Optional overall title
    if title:
        # place title slightly lower and leave room for it
        g.fig.suptitle(title, y=0.98)
        g.fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 4. Save as PDF
    # use tight bbox to ensure the suptitle is not clipped
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"Successfully saved plot to {output_filename}")


if __name__ == "__main__":
    
    # --- Usage Example ---
    files_to_plot = {
        "GCN Layer 1": r"Self Adaptive\SH\logs\raw\gcn_1.csv",
        "GCN Layer 2": r"Self Adaptive\SH\logs\raw\gcn_2.csv",
        "GCN Layer 3": r"Self Adaptive\SH\logs\raw\gcn_3.csv",
        "GCN Layer 5": r"Self Adaptive\SH\logs\raw\base.csv",
        "GCN Layer 7": r"Self Adaptive\SH\logs\raw\gcn_7.csv",
    }

    plot_learning_curves(files_to_plot,output_filename="Self Adaptive\SH\plots\gcn.pdf", title="GCN Layer Experiments Learning Curves")
