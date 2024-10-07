import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def readfile(file, dimensions=2, sep=" "):
    """
    Input file format: 3 columns ,seperated by ' '
    Column 1: element1
    Column 2: element2
    Column 3: distance between element1 and element2
    For example: (id > 0)
        1   2   0.6
        1   3   2.3
        2   3   1.4
    Return (dist,xxdist,ND,N)
    """
    print("Loading the file ...")
    xx = np.genfromtxt(file, delimiter=sep, names=["x", "y", "dist"], dtype="i8,i8,f8")
    # ND: number of data point
    X = xx["x"]
    Y = xx["y"]
    xxdist = xx["dist"]
    ND = Y.max()
    NL = X.max()
    if NL > ND:
        ND = NL
    # N: number of point pairs/distance
    N = xx.shape[0]
    dist = np.zeros((ND, ND))

    # dist may save half of memory
    for i in range(N):
        ii = X[i] - 1
        jj = Y[i] - 1
        dist[ii, jj] = xxdist[i]
        dist[jj, ii] = xxdist[i]

    return (dist, xxdist, xx, ND, N)


def import_cluster_assignation(path):
    # Find the file starting with CLUSTER_ASSIGNATION
    file = glob.glob(path)[0]
    print(file)
    # Read the file into a DataFrame
    df = pd.read_csv(
        file, delimiter="\t", header=None
    )  # Adjust delimiter and header as needed

    labels = df.iloc[:, 1].values
    halo_labels = df.iloc[:, 2].values

    return labels, halo_labels


def analyze_distance_distribution(xx):
    unique_points = np.unique(np.concatenate((xx["x"], xx["y"])))
    point_distributions = {}

    # Calculate distances for each point
    for point in unique_points:
        distances = xx["dist"][(xx["x"] == point) | (xx["y"] == point)]
        point_distributions[point] = distances

    zero_distance_counts = {}
    points_with_zero_distance = 0
    total_zero_distances = 0
    zero_distance_pairs = set()

    # Analyze zero distances
    for i in range(len(xx)):
        if xx["dist"][i] == 0:
            pair = (min(xx["x"][i], xx["y"][i]), max(xx["x"][i], xx["y"][i]))

            # Avoid counting duplicate pairs
            if pair not in zero_distance_pairs:
                zero_distance_pairs.add(pair)
                total_zero_distances += 1  # Count zero distance

                # Count zero distance occurrences for each point
                zero_distance_counts[xx["x"][i]] = (
                    zero_distance_counts.get(xx["x"][i], 0) + 1
                )
                zero_distance_counts[xx["y"][i]] = (
                    zero_distance_counts.get(xx["y"][i], 0) + 1
                )

    points_with_zero_distance = len(zero_distance_counts)

    sorted_zero_distance_counts = sorted(
        zero_distance_counts.items(), key=lambda item: item[1], reverse=True
    )

    return (
        point_distributions,
        sorted_zero_distance_counts,
        points_with_zero_distance,
        total_zero_distances,
    )


if __name__ == "__main__":
    # Example usage
    dist, xxdist, xx, ND, N = readfile(
        "/home/mikesteel/possession-chain-analysis/lcss_distance/eps_10/data_array.dat"
    )

    (
        point_distributions,
        sorted_zero_distance_counts,
        points_with_zero_distance,
        total_zero_distances,
    ) = analyze_distance_distribution(xx)

    # Print the points with the highest count of zero distances
    print("Points with the highest count of zero distances:")
    for point, count in sorted_zero_distance_counts[:10]:
        print(f"Point: {point}, Zero Distance Count: {count}")

    # Print the number of points that have at least one zero distance
    print(
        f"Number of points with at least one zero distance: {points_with_zero_distance}"
    )

    # Print the total number of zero distances
    print(f"Total number of zero distances: {total_zero_distances}")

    # Plot the distribution for the point with the highest count of zero distances
    point_to_plot = sorted_zero_distance_counts[0][0]
    distances_to_plot = point_distributions[point_to_plot]

    plt.hist(distances_to_plot, bins=30, density=True, alpha=0.6, color="g")
    kde = gaussian_kde(distances_to_plot)
    x = np.linspace(0, max(distances_to_plot), 1000)
    plt.plot(x, kde(x), "k", linewidth=2)
    plt.title(f"Distance Distribution for Point {point_to_plot}")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.show()
