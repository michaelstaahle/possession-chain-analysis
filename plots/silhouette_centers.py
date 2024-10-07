import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from itertools import combinations


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

    return (dist, xxdist, ND, N)


def rhodelta(dist, xxdist, ND, N, percent=2.0):
    """
    Input file format: 3 columns ,seperated by ' '
    Return (rho,delta,ordrho)
    """
    print("Caculating rho and delta...")
    print("average percentage of neighbours (hard coded): %5.6f" % (percent))

    position = int(round(N * percent / 100))
    sda = np.sort(xxdist)
    dc = sda[position]
    print("Computing Rho with gaussian kernel of radius: %12.6f\n" % (dc))

    rho = np.zeros(ND)
    # Gaussian kernel
    for i in range(ND - 1):
        for j in range((i + 1), ND):
            rho[i] = rho[i] + np.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
            rho[j] = rho[j] + np.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))

    maxd = dist.max()
    ordrho = (-rho).argsort()
    delta = np.zeros(ND)
    nneigh = np.zeros(ND, dtype=int)
    delta[ordrho[0]] = -1
    nneigh[ordrho[0]] = 0

    for ii in range(1, ND):
        delta[ordrho[ii]] = maxd
        for jj in range(ii):
            if dist[ordrho[ii], ordrho[jj]] < delta[ordrho[ii]]:
                delta[ordrho[ii]] = dist[ordrho[ii], ordrho[jj]]
                nneigh[ordrho[ii]] = ordrho[jj]

    delta[ordrho[0]] = delta.max()
    return (rho, delta, ordrho, dc, nneigh)


def plot_silhouette_scores_centers_perm(dist, ND, rho, delta, ordrho, nneigh, n):
    silhouette_scores = []
    highest_scores = {}

    # Calculate rho*delta and sort in descending order
    rho_delta = rho * delta
    sorted_indices = np.argsort(-rho_delta)

    max_clusters = 100
    for n_clusters in range(2, min(ND, max_clusters) + 1):
        # Generate all combinations of the top n rho*delta values
        if n_clusters > n:
            break
        top_indices = sorted_indices[:n]
        comb = list(combinations(top_indices, n_clusters))
        print(f"n_clusters: {n_clusters}, combinations: {len(comb)}")

        for c in comb:
            cl = np.zeros(ND, dtype=int) - 1
            icl = np.zeros(n_clusters, dtype=int)

            for i in range(n_clusters):
                idx = c[i]
                cl[idx] = i
                icl[i] = idx

            for i in range(ND):
                if cl[ordrho[i]] == -1:
                    cl[ordrho[i]] = cl[nneigh[ordrho[i]]]

            labels = cl + 1
            silhouette_avg = silhouette_score(dist, labels, metric="precomputed")
            silhouette_scores.append([n_clusters, silhouette_avg])

            if (
                n_clusters not in highest_scores
                or silhouette_avg > highest_scores[n_clusters][0]
            ):
                highest_scores[n_clusters] = (silhouette_avg, c)

    silhouette_scores_np = np.array(silhouette_scores)

    plt.plot(
        silhouette_scores_np[:, 0],
        silhouette_scores_np[:, 1],
        "o",
        label="Combinations",
    )

    highest_n_clusters = sorted(highest_scores.keys())
    highest_silhouette_scores = [highest_scores[nc][0] for nc in highest_n_clusters]

    plt.plot(
        highest_n_clusters,
        highest_silhouette_scores,
        "r-",
        label="Highest Scores",
    )

    plt.xlabel("Number of Cluster Centers")
    plt.ylabel("Silhouette Score")
    plt.title(
        r"Silhouette Score (based on combinations of points with the highest $\delta \cdot \rho$) vs Number of Cluster Centers"
    )
    plt.grid(True)
    plt.legend()
    plt.show()

    return highest_scores


def run(fi, sep=" ", percent=2.0, n=8):
    dist, xxdist, ND, N = readfile(fi, sep=sep)
    rho, delta, ordrho, dc, nneigh = rhodelta(dist, xxdist, ND, N, percent)
    highest_scores = plot_silhouette_scores_centers_perm(
        dist, ND, rho, delta, ordrho, nneigh, n
    )
    print(
        "Highest silhouette scores and corresponding points per number of cluster centers:"
    )
    for n_clusters, (score, points) in highest_scores.items():
        print(f"{n_clusters} clusters: {score}, points: {points}")


def silhouette_scores(dist, ND, rho, delta, ordrho, nneigh):
    # Calculate rho * delta for each point
    rho_delta = rho * delta
    rho_delta_order = np.argsort(
        -rho_delta
    )  # Sort indices by rho * delta in descending order

    # Store silhouette scores
    silhouette_scores = []

    # Maximum number of clusters to consider
    max_clusters = 100
    for n_clusters in range(
        2, min(ND, max_clusters) + 1
    ):  # Starting from 2 cluster centers
        # print(f"\nAssigning clusters with {n_clusters} centers")

        NCLUST = 0
        cl = (
            np.zeros(ND, dtype=int) - 1
        )  # Initialize cluster assignments (-1 means unassigned)
        icl = np.zeros(n_clusters, dtype=int)  # Cluster centers

        # Assign the top `n_clusters` points as cluster centers
        for i in range(n_clusters):
            idx = rho_delta_order[i]  # Get index of the i-th highest rho * delta point
            cl[idx] = NCLUST
            icl[NCLUST] = idx
            NCLUST += 1

        # Assign remaining points to clusters based on nearest higher-density neighbor
        for i in range(ND):
            if cl[ordrho[i]] == -1:  # If the point is unassigned
                cl[ordrho[i]] = cl[nneigh[ordrho[i]]]

        # Calculate silhouette score and store it
        labels = (
            cl + 1
        )  # Cluster labels for silhouette_score function (labels must be >= 1)
        silhouette_avg = silhouette_score(dist, labels, metric="precomputed")
        silhouette_scores.append([n_clusters, silhouette_avg])
        # print(f"Silhouette score for {n_clusters} clusters: {silhouette_avg}")

    silhouette_scores_np = np.array(silhouette_scores)

    # Return the silhouette scores list
    return silhouette_scores_np


def plot_silhouette_scores_centers(file, percent, sep=" "):
    dist, xxdist, ND, N = readfile(file, sep=sep)
    rho, delta, ordrho, dc, nneigh = rhodelta(dist, xxdist, ND, N, percent)
    silhouette_scores_np = silhouette_scores(dist, ND, rho, delta, ordrho, nneigh)
    print(silhouette_scores_np[0])
    # Plot silhouette scores
    plt.plot(silhouette_scores_np[:, 0], silhouette_scores_np[:, 1], "o-")
    plt.xlabel("Number of Cluster Centers")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs Number of Cluster Centers")
    # Set custom ticks
    # x_ticks = np.arange(0, max(silhouette_scores_np[:, 0]) + 5, 5)
    # y_ticks = np.arange(
    #    -0.2,
    #    0.2,
    #    0.02,
    # )
    # plt.xticks(x_ticks)
    # plt.yticks(y_ticks)
    plt.grid(True)
    plt.show()
