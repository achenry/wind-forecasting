import os

import polars as pl
import seaborn as sns

from ..utils.config import Config

# INFO: Currently not used
class Plotter:
    @staticmethod
    def plot_metrics(train_metrics, val_metrics, metric_names, dataset):
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        
        all_data = []
        for metric_name in metric_names:
            # Ensure train and val metrics have the same length
            min_len = min(len(train_metrics[metric_name]), len(val_metrics[metric_name]))
            epochs = list(range(1, min_len + 1))
            
            # Truncate metrics to the same length
            train_values = train_metrics[metric_name][:min_len]
            val_values = val_metrics[metric_name][:min_len]
            
            train_df = pl.DataFrame({
                'Epoch': epochs,
                'Value': train_values,
                'Type': ['Train'] * min_len,
                'Metric': [metric_name] * min_len
            })
            
            val_df = pl.DataFrame({
                'Epoch': epochs,
                'Value': val_values,
                'Type': ['Validation'] * min_len,
                'Metric': [metric_name] * min_len
            })
            
            all_data.extend([train_df, val_df])
        
        try:
            df = pl.concat(all_data)
            metric_data_pd = df.to_pandas()
            
            # Create FacetGrid
            g = sns.FacetGrid(metric_data_pd, col='Metric', col_wrap=len(metric_names), 
                            height=7, aspect=1.5)
            
            # Draw the lines
            g.map_dataframe(sns.lineplot, x='Epoch', y='Value', hue='Type',
                          palette=['#2ecc71', '#e74c3c'], linewidth=2.5)
            
            # Customize titles and labels
            g.set_titles(col_template="{col_name}", size=16, weight='bold', pad=20)
            g.set_axis_labels("Epoch", "Value", size=12)
            g.add_legend(title=None, fontsize=10)
            
            # Save the plot
            g.savefig(os.path.join(Config.PLOT_DIR, f'{dataset}_metrics_latest.png'),
                     bbox_inches='tight',
                     facecolor='white',
                     edgecolor='none',
                     dpi=300)
            
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            # Don't let plotting errors stop the training process
            pass
