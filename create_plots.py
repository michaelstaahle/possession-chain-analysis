from plots.plots import *
from helper_functions import *

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
