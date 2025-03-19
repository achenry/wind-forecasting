import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import colormaps
from gluonts.dataset.util import to_pandas

def highlight_entry(entry, color, ax, end_line=False):
    start = entry["start"].to_timestamp()
    # end = entry["start"] + (entry["target"].shape[1] * entry["start"].freq.delta)
    end = (entry["start"] + entry["target"].shape[1]).to_timestamp()
    # print(f"start time = {start}, end time = {end}")
    if end_line:
        ax.axvline(x=end)
    else:
        ax.axvspan(start, end, facecolor=color, alpha=0.2)
    
def plot_dataset_splitting(original_dataset, train_dataset, test_pairs, val_pairs, n_cgs, n_splits, n_output_vars):
    # n_output_vars = next(original_dataset)["target"].shape[0]
    
    fig, axs = plt.subplots(n_output_vars, 1, sharex=True)
    for o, original_entry in enumerate(original_dataset):
        df = to_pandas(original_entry, is_multivariate=True).reset_index(names="time")
        df["time"] = df["time"].dt.to_timestamp()
        for a in range(n_output_vars):
            sns.lineplot(data=df, ax=axs[a], x="time", y=df.columns[a + 1])
    
    # train_entries = islice(train_dataset, int(o * n_splits), int((o + 1) * n_splits))
    # for t, train_entry in enumerate(train_entries):
    # for t in range(n_splits):
        # train_entry = next(train_dataset)
    for train_entry in train_dataset:
        for a in range(n_output_vars):
            highlight_entry(train_entry, "red", ax=axs[a], end_line=True)
    # plt.show()
    axs[0].legend(["sub dataset", "training dataset"], loc="upper left")

    # for t in range(n_splits):
    #     test_input, test_label = next(test_pairs)
    # test_input_colors = cycle(colormaps["seismic"](range(n_splits * n_cgs)))
    # test_output_colors = cycle(colormaps["Spectral"](range(n_splits * n_cgs)))
    # val_input_colors = cycle(colormaps["PiYG"](range(n_splits * n_cgs)))
    # val_output_colors = cycle(colormaps["PuOr"](range(n_splits * n_cgs)))
    colors = colormaps["Pastel1"].colors
    for t, (test_input, test_label) in enumerate(test_pairs):
        for a in range(n_output_vars):
            highlight_entry(test_input, colors[0], axs[a])
            highlight_entry(test_label, colors[1], axs[a])
    for v, (val_input, val_label) in enumerate(val_pairs):
        for a in range(n_output_vars):
            highlight_entry(val_input, colors[2], axs[a])
            highlight_entry(val_label, colors[3], axs[a])
        plt.legend(["sub dataset", "test input", "test label"], loc="upper left")
    
    plt.show()