import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

from sklearn.metrics import silhouette_samples, silhouette_score


# Assuming data_array["distance"] is available
def plot_histogram(data_array, num_bins=100):
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    # Calculate weights for each instance to normalize the histogram
    weights = np.ones_like(data_array["distance"]) / len(data_array["distance"])
    counts, bins, patches = plt.hist(
        data_array["distance"],
        bins=num_bins,
        weights=weights,
        alpha=0.75,
        edgecolor="black",
    )

    # Calculate bin size
    bin_size = bins[1] - bins[0]

    plt.title("Distribution of distances between ball trajectories")
    plt.xlabel("Distance")
    plt.ylabel("Percentage of Data")
    plt.grid(True)

    # Display bin size on the plot
    plt.text(
        0.95,
        0.95,
        f"Bin size: {bin_size:.2f}",
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="top",
    )

    plt.show()


def plot_trajectory(trajectories):
    # Load the background picture
    background = mpimg.imread("plots/field_2023250_1280.png")

    # Plot the background
    plt.imshow(
        background, extent=[0, 100, 0, 100], aspect=0.5
    )  # Adjust extent as needed

    # Choose the 'Blues' colormap
    cmap = cm.Blues

    for trajectory in trajectories:
        # The total number of steps in the trajectory
        total_steps = len(trajectory) - 1

        for i in range(total_steps):
            # Normalize the current step to [0, 1]
            normalized_index = i / total_steps

            # Get the color from the colormap
            color = cmap(
                normalized_index + 0.2
            )  # Adding 0.2 to avoid too light colors, adjust as needed
            offset_x = random.uniform(-1, 1)  # Adjust the range as needed
            offset_y = random.uniform(-1, 1)  # Adjust the range as needed

            plt.arrow(
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

    plt.xlim(0, 100)  # Adjust as needed
    plt.ylim(0, 100)  # Adjust as needed
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
