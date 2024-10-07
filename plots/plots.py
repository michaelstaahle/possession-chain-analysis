import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

from sklearn.metrics import silhouette_samples, silhouette_score


# Assuming data_array is available
def plot_multiple_histograms(datasets, num_bins=100):
    """
    Plots multiple histograms in a single figure based on the provided datasets.

    Parameters:
    datasets (list): A list of dictionaries, each containing 'data' (array of values)
                     and 'title' (title for the corresponding histogram).
    num_bins (int): Number of bins for each histogram.
    """
    # Determine the number of subplots based on the number of datasets
    num_datasets = len(datasets)
    num_cols = 3  # Set number of columns for subplots
    num_rows = (
        num_datasets + num_cols - 1
    ) // num_cols  # Calculate required number of rows

    # Create subplots with appropriate size
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))

    # Flatten the axes array for easy indexing
    axs = axs.ravel() if num_datasets > 1 else [axs]

    for i, dataset in enumerate(datasets):
        data_array = dataset["data"]
        title = dataset["title"]

        # Calculate weights to normalize the histogram
        weights = np.ones_like(data_array) / len(data_array)

        # Plot histogram in the corresponding subplot
        counts, bins, patches = axs[i].hist(
            data_array,
            bins=num_bins,
            weights=weights,
            alpha=0.75,
            edgecolor="black",
        )

        # Calculate bin size
        bin_size = bins[1] - bins[0]

        # Set the title and labels
        axs[i].set_title(title)
        axs[i].set_xlabel("Distance")
        axs[i].set_ylabel("Percentage of Data")
        axs[i].grid(True)

        # Display bin size on the plot
        # axs[i].text(
        #     0.95,
        #     0.95,
        #     f"Bin size: {bin_size:.2f}",
        #     transform=axs[i].transAxes,
        #     horizontalalignment="right",
        #     verticalalignment="top",
        # )

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_trajectory_dict(trajectories_dict):
    # Load the background picture
    background = mpimg.imread("plots/field_2023250_1280.png")

    # Determine the number of subplots needed
    num_subplots = len(trajectories_dict)
    num_cols = 2  # Adjust the number of columns as needed
    num_rows = (num_subplots + num_cols - 1) // num_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Choose the 'Blues' colormap
    cmap = cm.Blues

    for idx, (label, trajectories) in enumerate(trajectories_dict.items()):
        ax = axes[idx]
        # Plot the background
        ax.imshow(background, extent=[-4, 104, -4, 104], aspect="auto")

        for trajectory in trajectories:
            # The total number of steps in the trajectory
            total_steps = len(trajectory) - 1

            for i in range(total_steps):
                # Normalize the current step to [0, 1]
                normalized_index = i / total_steps

                # Get the color from the colormap
                color = cmap(normalized_index + 0.2)  # Adjust as needed
                offset_x = random.uniform(-1, 1)  # Adjust the range as needed
                offset_y = random.uniform(-1, 1)  # Adjust the range as needed

                ax.arrow(
                    trajectory[i][0] + offset_x,
                    trajectory[i][1] + offset_y,
                    trajectory[i + 1][0] - trajectory[i][0],
                    trajectory[i + 1][1] - trajectory[i][1],
                    head_width=1.5,
                    head_length=2,
                    fc=color,
                    ec=color,
                    length_includes_head=True,
                )

        ax.set_xlim(-4, 104)  # Adjust as needed
        ax.set_ylim(-4, 104)  # Adjust as needed
        ax.set_title(f"Ball Trajectories in Cluster {label}")

    # Remove any unused subplots
    for idx in range(num_subplots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def plot_trajectory(
    trajectories, title=r"Example of two ball trajectories $T_a$ and $T_b$"
):
    # Load the background picture
    background = mpimg.imread("plots/field_2023250_1280.png")

    # Plot the background
    plt.imshow(
        background, extent=[-4, 104, -4, 104], aspect="auto"
    )  # Adjust extent as needed

    # Choose the colormaps
    cmap_a = cm.Blues
    cmap_b = cm.Reds

    # Plot the first trajectory with the 'Blues' colormap
    trajectory_a = trajectories[0]
    total_steps_a = len(trajectory_a) - 1

    for i in range(total_steps_a):
        normalized_index = i / total_steps_a
        color = cmap_a(normalized_index + 0.2)
        plt.arrow(
            trajectory_a[i][0],
            trajectory_a[i][1],
            trajectory_a[i + 1][0] - trajectory_a[i][0],
            trajectory_a[i + 1][1] - trajectory_a[i][1],
            head_width=1.5,
            head_length=2,
            fc=color,
            ec=color,
            length_includes_head=True,
        )
        plt.annotate(
            f"a{i+1}",
            (trajectory_a[i][0], trajectory_a[i][1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Annotate the last point of trajectory_a
    plt.annotate(
        f"a{total_steps_a+1}",
        (trajectory_a[-1][0], trajectory_a[-1][1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

    # Plot the second trajectory with the 'Reds' colormap
    trajectory_b = trajectories[1]
    total_steps_b = len(trajectory_b) - 1

    for i in range(total_steps_b):
        normalized_index = i / total_steps_b
        color = cmap_b(normalized_index + 0.2)
        plt.arrow(
            trajectory_b[i][0],
            trajectory_b[i][1],
            trajectory_b[i + 1][0] - trajectory_b[i][0],
            trajectory_b[i + 1][1] - trajectory_b[i][1],
            head_width=1.5,
            head_length=2,
            fc=color,
            ec=color,
            length_includes_head=True,
        )
        plt.annotate(
            f"b{i+1}",
            (trajectory_b[i][0], trajectory_b[i][1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Annotate the last point of trajectory_b
    plt.annotate(
        f"b{total_steps_b+1}",
        (trajectory_b[-1][0], trajectory_b[-1][1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

    plt.title(title)
    plt.xlim(-4, 104)  # Adjust as needed
    plt.ylim(-4, 104)  # Adjust as needed
    plt.show()


def silhouette_plot(D, labels, range_n_clusters):

    for n_clusters in range_n_clusters:
        silhouette_avg = silhouette_score(D, labels, metric="precomputed")
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(D, labels, metric="precomputed")

        silhouette_avg, sample_silhouette_values
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(D) + (n_clusters + 1) * 10])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        y_lower = 10
        for i in range(1, n_clusters + 1):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()


def plot_decision_graphs():
    # Load the data from the files
    file_02 = "frechet_distance/results/2_centers/DECISION_GRAPH_0.2.txt"
    file_016 = "frechet_distance/results/2_centers/DECISION_GRAPH_0.16.txt"
    file_009 = "frechet_distance/results/2_centers/DECISION_GRAPH_0.09.txt"

    data_02 = pd.read_csv(file_02, sep=" ", header=None)
    data_016 = pd.read_csv(file_016, sep=" ", header=None)
    data_009 = pd.read_csv(file_009, sep=" ", header=None)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Set font size and font family
    plt.rcParams.update({"font.size": 12, "font.family": "Times New Roman"})

    # Plot each dataset in a subplot
    axs[0].scatter(data_02[0], data_02[1], color="blue")
    axs[0].set_title(r"Scatter plot of ρ vs δ with $d_c = 0.2$")
    axs[0].set_xlabel(r"$\rho$")
    axs[0].set_ylabel(r"$\delta$")
    axs[0].grid(True)
    axs[0].set_ylim([0, 0.5])  # Set y-axis range

    axs[1].scatter(data_016[0], data_016[1], color="green")
    axs[1].set_title(r"Scatter plot of ρ vs δ with $d_c = 0.16$")
    axs[1].set_xlabel(r"$\rho$")
    axs[1].set_ylabel(r"$\delta$")
    axs[1].grid(True)
    axs[1].set_ylim([0, 0.5])  # Set y-axis range

    axs[2].scatter(data_009[0], data_009[1], color="red")
    axs[2].set_title(r"Scatter plot of ρ vs δ with $d_c = 0.09$")
    axs[2].set_xlabel(r"$\rho$")
    axs[2].set_ylabel(r"$\delta$")
    axs[2].grid(True)
    axs[2].set_ylim([0, 0.5])  # Set y-axis range

    # Set a title for the whole figure
    fig.suptitle(r"Decission Graphs for Different Values of $d_c$")

    # Display the figure
    plt.tight_layout()
    plt.show()


def plot_single_decision_graph(file_path):
    # Load the data from the file
    data = pd.read_csv(file_path, sep=" ", header=None)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Set font size and font family
    plt.rcParams.update({"font.size": 12, "font.family": "Times New Roman"})

    # Plot the original decision graph
    axs[0].scatter(data[0], data[1], color="black")
    axs[0].set_title(r"Scatter plot of ρ vs δ")
    axs[0].set_xlabel(r"$\rho$")
    axs[0].set_ylabel(r"$\delta$")
    axs[0].grid(True)
    axs[0].set_ylim([0, 0.5])  # Set y-axis range

    # Calculate delta*rho and sort in descending order
    delta_rho = data[0] * data[1]
    sorted_indices = delta_rho.sort_values(ascending=False).index
    sorted_delta_rho = delta_rho[sorted_indices].reset_index(drop=True)

    # Plot delta*rho in descending order
    axs[1].scatter(range(len(sorted_delta_rho)), sorted_delta_rho, color="black")
    axs[1].set_title(r"Delta*Rho in Descending Order")
    axs[1].set_xlabel(r"Index")
    axs[1].set_ylabel(r"$\delta \cdot \rho$")
    axs[1].grid(True)

    # Mark the top 4 points with highest delta*rho
    top_indices = sorted_indices[:4]
    colors = ["red", "green", "blue", "orange"]

    for i, idx in enumerate(top_indices):
        # Mark the points in the original decision graph
        axs[0].scatter(
            data[0][idx],
            data[1][idx],
            color=colors[i],
            edgecolor="black",
            s=100,
            zorder=5,
        )
        # Mark the points in the delta*rho plot
        axs[1].scatter(
            sorted_indices.get_loc(idx),
            sorted_delta_rho[sorted_indices.get_loc(idx)],
            color=colors[i],
            edgecolor="black",
            s=100,
            zorder=5,
        )

    # Set a title for the whole figure
    fig.suptitle(r"Decision Graph and Delta*Rho")

    # Display the figure
    plt.tight_layout()
    plt.show()


# Example usage
# file_path = "frechet_distance/results/2_centers/DECISION_GRAPH_0.2.txt"
# plot_single_decision_graph(file_path)


def plot_decision_graphs_2(datasets):
    """
    Plots decision graphs for given datasets in a grid where varying d_c values
    are on the horizontal axis and varying eps values are on the vertical axis.
    Both x-axes and y-axes are independent for all subplots, and y-limits are automatic.

    Parameters:
    datasets (list): A list of dictionaries, each containing 'eps', 'd_c', and 'file_path'.

    Example:
    datasets = [
        {"eps": 0.2, "d_c": 0.2, "file_path": "path_to_file1.txt"},
        {"eps": 0.2, "d_c": 0.16, "file_path": "path_to_file2.txt"},
        {"eps": 0.2, "d_c": 0.09, "file_path": "path_to_file3.txt"},
        {"eps": 0.16, "d_c": 0.2, "file_path": "path_to_file4.txt"},
        {"eps": 0.16, "d_c": 0.16, "file_path": "path_to_file5.txt"},
        {"eps": 0.16, "d_c": 0.09, "file_path": "path_to_file6.txt"},
        {"eps": 0.09, "d_c": 0.2, "file_path": "path_to_file7.txt"},
        {"eps": 0.09, "d_c": 0.16, "file_path": "path_to_file8.txt"},
        {"eps": 0.09, "d_c": 0.09, "file_path": "path_to_file9.txt"}
    ]
    plot_decision_graphs(datasets)
    """
    # Load data from files and collect unique eps and d_c values
    data_dict = {}
    eps_values = set()
    d_c_values = set()
    for dataset in datasets:
        eps = dataset["eps"]
        d_c = dataset["d_c"]
        data = pd.read_csv(dataset["file_path"], sep=" ", header=None)
        data_dict[(eps, d_c)] = data
        eps_values.add(eps)
        d_c_values.add(d_c)

    # Sort eps and d_c values
    eps_values = sorted(eps_values, reverse=True)
    d_c_values = sorted(d_c_values, reverse=True)

    # Create subplots without sharing axes
    num_rows = len(eps_values)
    num_cols = len(d_c_values)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 6))

    # Set font size and font family
    plt.rcParams.update({"font.size": 12, "font.family": "Times New Roman"})

    # Plot each dataset in the correct subplot
    colors = ["blue", "green", "red"]
    for i, eps in enumerate(eps_values):
        for j, d_c in enumerate(d_c_values):
            # Handle cases where axs is 1D
            if num_rows == 1 and num_cols == 1:
                ax = axs
            elif num_rows == 1:
                ax = axs[j]
            elif num_cols == 1:
                ax = axs[i]
            else:
                ax = axs[i][j]
            key = (eps, d_c)
            if key in data_dict:
                data = data_dict[key]
                ax.scatter(data[0], data[1], color=colors[j])
                title = f"$d_c = {d_c}$, $\epsilon = {eps}$"
                ax.set_title(title)
                ax.set_xlabel(r"$\rho$")
                ax.set_ylabel(r"$\delta$")
                ax.grid(True)
                # Removed ax.set_ylim([0, 0.5]) to allow automatic y-limits
            else:
                ax.set_visible(False)  # Hide subplot if data not available

    # Adjust spacing between subplots
    # plt.subplots_adjust(wspace=0.4, hspace=0.6)

    # Set a title for the whole figure
    fig.suptitle(
        r"Decision Graphs for Different Values of $d_c$",
        fontsize=16,
        y=0.99,
    )

    # Display the figure
    plt.show()
