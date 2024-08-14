import matplotlib.pyplot as plt
import numpy as np


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
