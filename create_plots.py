from helper_functions import *
from plots.plots import *
from plots.silhouette_centers import plot_silhouette_scores_centers
from possession_chains import possession_chains


def create_silhouette_plot():
    labels, halo_labels = import_cluster_assignation(
        "frechet_distance/results/2_centers/CLUSTER_ASSIGNATION_12.62_0.26_.txt"
    )
    dist, xxdist, ND, N = readfile("frechet_distance/data_array.dat")
    # Filter labels to only include labels > 0
    filtered_indices = np.where(halo_labels > 0)[0]
    filtered_labels = labels[filtered_indices]

    # Filter the distance matrix to only include rows and columns corresponding to the filtered labels
    filtered_dist = dist[np.ix_(filtered_indices, filtered_indices)]
    print(len(filtered_labels))
    print(filtered_dist.shape)
    silhouette_plot(filtered_dist, filtered_labels, [2])


def plot_clusters(assignation_path, halos=True):
    print("Import possession chains")
    pc = possession_chains()
    print("Create ball trajectories")
    ball_trajectories, ball_trajectories_dfs = pc.create_ball_trajectories()

    labels, halo_labels = import_cluster_assignation(assignation_path)

    if halos:
        labels = halo_labels

    unique_labels = np.unique(labels[labels > 0])
    # Store the ball_trajectories for different label values in a dictionary
    ball_trajectories_dict = {}
    ball_trajectories_dfs_dict = {}
    goals_dict = {}

    for label in unique_labels:
        # Filter indices for the current label
        filtered_indices = np.where(labels == label)[0]
        # Filter ball_trajectories and ball_trajectories_dfs based on those indices
        filtered_ball_trajectories = [ball_trajectories[i] for i in filtered_indices]
        filtered_ball_trajectories_dfs = [
            ball_trajectories_dfs[i] for i in filtered_indices
        ]
        # Store in dictionaries
        ball_trajectories_dict[label] = filtered_ball_trajectories
        ball_trajectories_dfs_dict[label] = filtered_ball_trajectories_dfs

        # Count goals for the current label
        goals = 0
        for df in filtered_ball_trajectories_dfs:
            if "Goal" in df.description.values:
                goals += 1
        goals_dict[label] = goals

        # Print the goals for each label
    for label, goals in goals_dict.items():
        print(f"Label {label}: {goals} goals")

    plot_trajectory_dict(ball_trajectories_dict)


def plot_lcss_decsicion_graphs():
    datasets = [
        #        {
        #            "eps": 10,
        #            "d_c": 0.33,
        #            "file_path": "lcss_distance/eps_10/results/perc_2_d_c_0.33/DECISION_GRAPH.txt",
        #        },
        #        {
        #            "eps": 10,
        #            "d_c": 0.25,
        #            "file_path": "lcss_distance/eps_10/results/perc_1_d_c_0.25/DECISION_GRAPH.txt",
        #        },
        #        {
        #            "eps": 10,
        #            "d_c": 0.16,
        #            "file_path": "lcss_distance/eps_10/results/perc_0.5_d_c_0.16/DECISION_GRAPH.txt",
        #        },
        # {
        #    "eps": 7.5,
        #    "d_c": 0.33,
        #    "file_path": "lcss_distance/eps_7.5/results/perc_0.5_d_c_0.33/DECISION_GRAPH.txt",
        # },
        # {
        #    "eps": 7.5,
        #    "d_c": 0.25,
        #    "file_path": "lcss_distance/eps_7.5/results/perc_0.3_d_c_0.25/DECISION_GRAPH.txt",
        # },
        # {
        #    "eps": 7.5,
        #    "d_c": 0.16,
        #    "file_path": "lcss_distance/eps_7.5/results/perc_0.1_d_c_0.16/DECISION_GRAPH.txt",
        # },
        #        {
        #            "eps": 5,
        #            "d_c": 0.33,
        #            "file_path": "lcss_distance/eps_5/results/perc_0.1_d_c_0.33/DECISION_GRAPH.txt",
        #        },
        #        {
        #            "eps": 5,
        #            "d_c": 0.25,
        #            "file_path": "lcss_distance/eps_5/results/perc_0.05_d_c_0.25/DECISION_GRAPH.txt",
        #        },
        #        {
        #            "eps": 5,
        #            "d_c": 0.16,
        #            "file_path": "lcss_distance/eps_5/results/perc_0.01_d_c_0.16/DECISION_GRAPH.txt",
        #        },
        {
            "eps": 20,
            "d_c": 0.2,
            "file_path": "lcss_distance/eps_20/results/perc_10_d_c_0.2/DECISION_GRAPH.txt",
        },
        {
            "eps": 20,
            "d_c": 0.16,
            "file_path": "lcss_distance/eps_20/results/perc_7.5_d_c_0.16/DECISION_GRAPH.txt",
        },
        {
            "eps": 20,
            "d_c": 0.12,
            "file_path": "lcss_distance/eps_20/results/perc_5_d_c_0.12/DECISION_GRAPH.txt",
        },
    ]

    plot_decision_graphs_2(datasets)


def create_lcss_distance_histogram():
    _, xxdist_5, _, _, _ = readfile("lcss_distance/eps_5/data_array.dat")
    _, xxdist_10, _, _, _ = readfile("lcss_distance/eps_10/data_array.dat")
    _, xxdist_20, _, _, _ = readfile("lcss_distance/eps_20/data_array.dat")
    data_sets = [
        {
            "data": xxdist_20,
            "title": r"Distribution of LCSS distance for $\epsilon = 20$",
        },
        {
            "data": xxdist_10,
            "title": r"Distribution of LCSS distance for $\epsilon = 10$",
        },
        {
            "data": xxdist_5,
            "title": r"Distribution of LCSS distance for $\epsilon = 5$",
        },
    ]
    plot_multiple_histograms(data_sets, 25)


if __name__ == "__main__":
    plot_silhouette_scores_centers(
        "/home/mikesteel/possession-chain-analysis/lcss_distance/eps_5/data_array.dat",
        0.1,
    )
