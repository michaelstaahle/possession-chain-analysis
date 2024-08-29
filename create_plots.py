from plots.plots import *
from helper_functions import *

labels = import_cluster_assignation("frechet_distance/results/10_centers")
dist, xxdist, ND, N = readfile("frechet_distance/data_array.dat")
silhouette_plot(dist, labels, [10])
