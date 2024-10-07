import numpy as np
import matplotlib.pyplot as plt
from traj_dist.distance import frechet
from scipy.interpolate import interp1d


# Function to compute the distance from a point to a line segment
def point_to_segment_distance(point, seg_start, seg_end):
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)
    seg_vec = seg_end - seg_start
    pt_vec = point - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq == 0:
        # Segment is a point
        closest_point = seg_start
        distance = np.linalg.norm(point - seg_start)
        return closest_point, distance
    t = np.dot(pt_vec, seg_vec) / seg_len_sq
    if t < 0:
        closest_point = seg_start
    elif t > 1:
        closest_point = seg_end
    else:
        closest_point = seg_start + t * seg_vec
    distance = np.linalg.norm(point - closest_point)
    return closest_point, distance


# Define the trajectories
a = np.array([[1, 2], [0, 4], [9, 2]])
b = np.array([[0, 1], [4, 0], [5, 1], [9, 1]])
# Convert to float64
a = a.astype(np.float64)
b = b.astype(np.float64)

a *= 4
b *= 4
a += 30
b += 30

# Compute the continuous Fréchet distance using traj_dist
fd = frechet(a, b)
print(f"Continuous Fréchet Distance: {fd:.6f}")

# Compute distances between each vertex of b and each edge of a
vertex_edge_distances = []

for i in range(len(b)):
    point_b = b[i]
    for j in range(len(a) - 1):
        seg_a_start = a[j]
        seg_a_end = a[j + 1]
        closest_point, distance = point_to_segment_distance(
            point_b, seg_a_start, seg_a_end
        )
        vertex_edge_distances.append(
            {"distance": distance, "point_a": closest_point, "point_b": point_b}
        )

# Compute distances between each vertex of a and each edge of b
for i in range(len(a)):
    point_a = a[i]
    for j in range(len(b) - 1):
        seg_b_start = b[j]
        seg_b_end = b[j + 1]
        closest_point, distance = point_to_segment_distance(
            point_a, seg_b_start, seg_b_end
        )
        vertex_edge_distances.append(
            {"distance": distance, "point_a": point_a, "point_b": closest_point}
        )

# Find the pair with distance closest to the computed Fréchet distance
distances = [item["distance"] for item in vertex_edge_distances]
distance_diffs = [abs(d - fd) for d in distances]
min_index = np.argmin(distance_diffs)
max_distance = distances[min_index]
max_pair = vertex_edge_distances[min_index]
point_a_fd = max_pair["point_a"]
point_b_fd = max_pair["point_b"]

# Prepare the figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

### Plot 1: LCSS Distance ###
ax = axs[0]

# Plot trajectory a in green with solid lines
ax.plot(
    a[:, 0], a[:, 1], color="green", linestyle="-", marker="o", label="Trajectory a"
)

# Annotate points a_i
for i, (x, y) in enumerate(a):
    ax.annotate(
        f"a{i+1}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )

# Plot trajectory b in blue with solid lines
ax.plot(b[:, 0], b[:, 1], color="blue", linestyle="-", marker="s", label="Trajectory b")

# Annotate points b_i
for i, (x, y) in enumerate(b):
    ax.annotate(
        f"b{i+1}", (x, y), textcoords="offset points", xytext=(0, -15), ha="center"
    )

# Set epsilon value for LCSS
epsilon = 10

# Calculate matching points based on epsilon
matching_pairs = []
for i, a_point in enumerate(a):
    for j, b_point in enumerate(b):
        distance = np.linalg.norm(a_point - b_point)
        if distance <= epsilon:
            matching_pairs.append((i, j))  # Store indices of matching points

# Show matching points with red dotted lines
label_added = False
for i, j in matching_pairs:
    if not label_added:
        ax.plot(
            [a[i][0], b[j][0]],
            [a[i][1], b[j][1]],
            "r--",
            linewidth=1,
            label="Matching points",
        )
        label_added = True
    else:
        ax.plot(
            [a[i][0], b[j][0]],
            [a[i][1], b[j][1]],
            "r--",
            linewidth=1,
        )

# Add a dummy plot for epsilon in the legend
ax.plot([], [], " ", label=r"$\epsilon$ =" + f"{epsilon}")

# Calculate LCSS length
lcss_length = len(matching_pairs)

# Calculate LCSS distance
lcss_distance = 1 - lcss_length / min(len(a), len(b))

# Display LCSS distance on the plot
ax.set_title(f"LCSS Distance = {lcss_distance:.2f}")
ax.legend()
ax.grid(True)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.axis("equal")

### Plot 2: Continuous Fréchet Distance ###
ax = axs[1]

# Plot trajectory a in green with solid lines
ax.plot(
    a[:, 0], a[:, 1], color="green", linestyle="-", marker="o", label="Trajectory a"
)

# Annotate points a_i
for i, (x, y) in enumerate(a):
    ax.annotate(
        f"a{i+1}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
    )

# Plot trajectory b in blue with solid lines
ax.plot(b[:, 0], b[:, 1], color="blue", linestyle="-", marker="s", label="Trajectory b")

# Annotate points b_i
for i, (x, y) in enumerate(b):
    ax.annotate(
        f"b{i+1}", (x, y), textcoords="offset points", xytext=(0, -15), ha="center"
    )

# Plot the Fréchet distance
ax.plot(
    [point_a_fd[0], point_b_fd[0]],
    [point_a_fd[1], point_b_fd[1]],
    "r--",
    linewidth=2,
    label="Fréchet Distance",
)

# Plot sample distances (grey dashed lines)
num_lines = 6
sample_distances = []
for idx in range(num_lines):
    # Choose points along the coupling path
    # For simplicity, select evenly spaced indices from vertex_edge_distances (excluding the max distance)
    if idx >= len(vertex_edge_distances):
        break
    item = vertex_edge_distances[idx]
    pa = item["point_a"]
    pb = item["point_b"]
    if item["distance"] <= fd:
        ax.plot(
            [pa[0], pb[0]], [pa[1], pb[1]], color="grey", linestyle="--", linewidth=1
        )
        sample_distances.append(item["distance"])

# Add a legend entry for the grey dashed lines
ax.plot([], [], color="grey", linestyle="--", linewidth=1, label="Sample Distances")

# Display the Fréchet distance on the plot
ax.set_title(f"Continuous Fréchet Distance = {fd:.3f}")
ax.legend()
ax.grid(True)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.axis("equal")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
