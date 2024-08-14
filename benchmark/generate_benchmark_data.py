import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def generate_benchmark_data(num_trajectories, mean_num_steps):
    distributions = ["uniform", "normal", "exponential"]
    data = []
    for _ in range(num_trajectories):
        num_steps = int(np.random.normal(mean_num_steps))
        num_steps = max(
            1, min(num_steps, 40)
        )  # Ensure num_steps is within the desired range
        distribution = np.random.choice(distributions)
        if distribution == "uniform":
            trajectory = np.random.uniform(low=0, high=100, size=(num_steps, 2))
            trajectory = np.clip(
                trajectory, 0, 100
            )  # Restrict points to be between 0 and 100
        elif distribution == "normal":
            trajectory = np.random.normal(loc=50, scale=25, size=(num_steps, 2))
            trajectory = np.clip(
                trajectory, 0, 100
            )  # Restrict points to be between 0 and 100
        elif distribution == "exponential":
            trajectory = np.random.exponential(scale=50, size=(num_steps, 2))
            trajectory = np.clip(
                trajectory, 0, 100
            )  # Restrict points to be between 0 and 100
        else:
            raise ValueError("Invalid distribution specified.")
        data.append(trajectory)
    return data


# num_trajectories = 2000
# mean_num_steps = 15
# benchmark_data = generate_benchmark_data(num_trajectories, mean_num_steps)
# print(benchmark_data)


def generate_advanced_benchmark_data(num_trajectories, mean_num_steps):
    distributions = ["bivariate_normal", "laplace", "logistic"]
    cluster_centers = {
        "bivariate_normal": [([50, 50],)],  # One center for bivariate_normal
        "laplace": [([50, 50],), ([30, 70],)],  # One center for laplace
        "logistic": [([70, 30],), ([50, 50],)],  # One center for logistic
    }
    data = []
    for _ in range(num_trajectories):
        num_steps = int(np.random.normal(mean_num_steps))
        num_steps = max(
            1, min(num_steps, 40)
        )  # Ensure num_steps is within the desired range
        distribution = np.random.choice(distributions)
        cluster_index = np.random.choice(len(cluster_centers[distribution]))
        centers = cluster_centers[distribution][cluster_index]
        center = centers[0]  # Directly use the first (and only) center

        if distribution == "bivariate_normal":
            mean = center
            cov = [[25, 0], [0, 25]]  # Reduced covariance for tighter clusters
            trajectory = np.random.multivariate_normal(mean, cov, num_steps)
        elif distribution == "laplace":
            loc = center
            scale = 10  # Reduced scale for tighter clusters
            trajectory = np.random.laplace(loc, scale, (num_steps, 2))
        elif distribution == "logistic":
            loc = center
            scale = 10  # Reduced scale for tighter clusters
            trajectory = np.random.logistic(loc, scale, (num_steps, 2))

        trajectory = np.clip(
            trajectory, 0, 100
        )  # Restrict points to be between 0 and 100
        data.append(trajectory)
    return data


# num_trajectories = 2000
# mean_num_steps = 15
# advanced_benchmark_data = generate_advanced_benchmark_data(
#     num_trajectories, mean_num_steps
# )
# print(advanced_benchmark_data)


def generate_clustered_samples(num_samples):
    # Adjusted cluster centers for greater separation and reduced variance
    cluster_centers = {
        "bivariate_normal": [(10, 10), (90, 90)],
        "laplace": [(10, 90), (90, 10)],
        "logistic": [(50, 50)],
    }
    data = []
    labels = []  # Keep track of distribution type for coloring
    for _ in range(num_samples):
        distribution = np.random.choice(list(cluster_centers.keys()))
        center = cluster_centers[distribution][
            np.random.randint(len(cluster_centers[distribution]))
        ]
        if distribution == "bivariate_normal":
            sample = np.random.multivariate_normal(center, [[20, 0], [0, 20]], 1)
        elif distribution == "laplace":
            sample = np.random.laplace(center, [5, 5], (1, 2))
        elif distribution == "logistic":
            sample = np.random.logistic(center, [5, 5], (1, 2))
        data.append(sample.flatten())
        labels.append(distribution)
    return data, labels


def plot_clustered_samples():
    num_samples = 2000
    clustered_samples_data, labels = generate_clustered_samples(num_samples)
    clustered_samples_array = np.array(clustered_samples_data)

    plt.figure(figsize=(10, 6))
    # Plot each distribution type in a different color
    colors = {"bivariate_normal": "red", "laplace": "blue", "logistic": "green"}
    for distribution in set(labels):
        idx = [i for i, label in enumerate(labels) if label == distribution]
        plt.scatter(
            clustered_samples_array[idx, 0],
            clustered_samples_array[idx, 1],
            alpha=0.5,
            label=distribution,
            color=colors[distribution],
        )
    plt.title("Clustered Samples from Various Distributions")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

    return clustered_samples_array, labels


clustered_samples_array, labels = plot_clustered_samples()


def create_pairwise_distance_array(point_data_array):
    # Calculate the pairwise Euclidean distances
    pairwise_distances = distance.pdist(point_data_array, "euclidean")
    # Convert the distances to a square matrix form
    distance_matrix = distance.squareform(pairwise_distances)

    # Initialize an empty list to store the 3-dimensional data
    data = []

    # Iterate over the distance matrix to get the indices and distance
    for i in range(distance_matrix.shape[0]):
        for j in range(
            i + 1, distance_matrix.shape[1]
        ):  # Avoid duplicate pairs and self-pairs
            # Increment indices by 1 to start from 1 instead of 0
            data.append((i + 1, j + 1, distance_matrix[i, j]))

    # Convert the list to a 3-dimensional NumPy array
    data_array = np.array(
        data, dtype=[("point_i", int), ("point_j", int), ("distance", float)]
    )

    np.savetxt("data_array.dat", data_array, fmt="%d %d %.5f")
    # data_3d_array now contains the desired 3-dimensional data with indices starting from 1
    return data_array
