import traj_dist.distance as tdist
import numpy as np

# ball_trajectory1 = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
# ball_trajectory2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
# print(tdist.sspd(ball_trajectory1, ball_trajectory2))


class distance_measures:
    def __init__(self, ball_trajectories):
        self.ball_trajectories = ball_trajectories
        self.n_ball_trajectories = len(ball_trajectories)

    def calculate_distance_measures(self, distance_measure="sspd", **kwargs):
        print(f"Calculating {distance_measure} distances")
        pdist = tdist.pdist(self.ball_trajectories, metric=distance_measure, **kwargs)
        print("Done")

        # Initialize an empty list to store the new data
        data = []

        # Counter for the distances array
        counter = 0

        # Iterate over the range of trajectories
        for i in range(self.n_ball_trajectories):
            for j in range(i + 1, self.n_ball_trajectories):
                # Append a tuple to the list that includes the outer loop index, the inner loop index, and the corresponding distance
                data.append((i, j, pdist[counter]))
                counter += 1

        # Convert the list to a numpy array
        data_array = np.array(
            data,
            dtype=[("trajectory_1", int), ("trajectory_2", int), ("distance", float)],
        )

        data_array["distance"] = np.around(data_array["distance"], 3)

        return data_array

    def save_distance(self, data_array):
        np.savetxt("data_array.dat", data_array, fmt="%d %d %.3f")

    def distances_statistics(self, data_array):
        # Calculate the mean of the distances
        mean = np.mean(data_array["distance"])

        # Calculate the standard deviation of the distances
        std = np.std(data_array["distance"])

        # Calculate the minimum distance
        min_distance = np.min(data_array["distance"])

        # Calculate the maximum distance
        max_distance = np.max(data_array["distance"])

        # Calculate the 90th percentile
        p90 = np.percentile(data_array["distance"], 90)
        # Calculate the 95th percentile
        p95 = np.percentile(data_array["distance"], 95)
        # Calculate the 99th percentile
        p99 = np.percentile(data_array["distance"], 99)

        return f"mean: {mean}, std: {std}, min_distance: {min_distance}, max_distance: {max_distance}, p90: {p90}, p95: {p95}, p99: {p99}"

    def set_outliers(self, data_array, threshold=None):
        if not threshold:
            threshold = 100 * np.median(data_array["distance"])

        outliers = data_array["distance"] > threshold
        data_array["distance"][outliers] = threshold

        return data_array


if __name__ == "__main__":
    from possession_chains import possession_chains

    print("Import possession chains")
    pc = possession_chains()
    print("Create ball trajectories")
    ball_trajectories, ball_trajectories_dfs = pc.create_ball_trajectories()
    print("Done")
    dm = distance_measures(ball_trajectories)
