import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_learning_curves(file_map_or_path, output_filename="learning_curves.pdf", title=None):
    """
    Plots train and validation loss from multiple CSV files.
    
    Parameters:
    - file_map: Dictionary where keys are experiment names (e.g., 'dilation_1') 
                and values are file paths (e.g., 'dilation_1.csv').
    - output_filename: Name of the PDF file to save.
    """
    
    # 1. Load data
    sns.set_theme(style="whitegrid")

    # If a dict was passed, keep the previous multi-file behavior
    if isinstance(file_map_or_path, dict):
        dfs = []
        for name, file in file_map_or_path.items():
            df = pd.read_csv(file)
            df['experiment'] = name
            dfs.append(df)
        combined_df = pd.concat(dfs)

        # reshape for FacetGrid
        df_melted = combined_df.melt(
            id_vars=['Step', 'experiment'],
            value_vars=['train_loss', 'val_loss'],
            var_name='Loss Type',
            value_name='Loss Value'
        )

        # Use FacetGrid to separate Train and Val loss into two charts
        g = sns.FacetGrid(df_melted, col="Loss Type", hue="experiment",
                          height=5, aspect=1.5, sharey=False)
        g.map(sns.lineplot, "Step", "Loss Value")
        g.add_legend(title="Experiment")
        g.set_titles("{col_name}")
        g.set_axis_labels("Step", "Loss")

        # Optional overall title
        if title:
            g.fig.suptitle(title, y=0.98)
            g.fig.tight_layout(rect=[0, 0, 1, 0.95])

    else:
        # single-file behavior: overlay train and val on the same axes
        file_path = file_map_or_path
        df = pd.read_csv(file_path)

        # melt to tidy format with Loss Type and Loss Value
        df_melted = df.melt(
            id_vars=['Step'],
            value_vars=['train_loss', 'val_loss'],
            var_name='Loss Type',
            value_name='Loss Value'
        )

        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(data=df_melted, x='Step', y='Loss Value', hue='Loss Type')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        if title:
            ax.set_title(title)
    
    # 4. Save as PDF
    # use tight bbox to ensure titles/labels are not clipped
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"Successfully saved plot to {output_filename}")


if __name__ == "__main__":
    
    file_map = r"Self Adaptive\SH\logs\raw\base.csv"
    plot_learning_curves(file_map_or_path=file_map,output_filename="Self Adaptive\SH\plots\learning_curve.pdf", title="Learning Curve")
