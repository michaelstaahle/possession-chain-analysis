import numpy as np
import matplotlib.pyplot as plt
import traj_dist.distance as tdist


# Define trajectories (e.g., a ball trajectory)
x_A = np.array([0, 20, 40, 60, 80, 100], dtype=np.float64)  # Convert to float64
y_A = np.array([0, 15, 30, 45, 30, 10], dtype=np.float64)  # Convert to float64

x_B = np.array(
    [0, 15, 30, 45, 55, 65, 70, 80, 90, 100], dtype=np.float64
)  # Convert to float64
y_B = np.array(
    [0, 10, 25, 35, 40, 45, 35, 25, 15, 10], dtype=np.float64
)  # Convert to float64

trajectory_A = np.array(list(zip(x_A, y_A)), dtype=np.float64)  # Ensure float64 dtype
trajectory_B = np.array(list(zip(x_B, y_B)), dtype=np.float64)  # Ensure float64 dtype


# Fréchet Distance using traj_dist
frechet_distance = tdist.frechet(trajectory_A, trajectory_B)

# LCSS Distance using traj_dist (with a threshold)
lcss_threshold = 10.0  # You can adjust this threshold based on your needs
lcss_distance = tdist.lcss(trajectory_A, trajectory_B, eps=lcss_threshold)

# Euclidean Distance using traj_dist (sliding window not necessary as it handles it internally)
euclidean_distance = tdist.sliding_window_euclidean_distance(
    trajectory_A, trajectory_B
)  # Hausdorff is similar to Euclidean

# Plot trajectories with distances in the legend
plt.figure(figsize=(10, 6))
plt.plot(x_A, y_A, "o-", label="Trajectory A")
plt.plot(x_B, y_B, "s-", label="Trajectory B")
plt.title("Trajectories A and B with Distance Measures")
plt.xlabel("X (Field Length)")
plt.ylabel("Y (Height)")
plt.grid(True)

# Add the distances to the legend
legend_text = (
    f"Fréchet Distance: {frechet_distance:.2f}\n"
    f"LCSS Distance: {lcss_distance:.2f}\n"
    f"Euclidean Distance (sliding window): {euclidean_distance:.2f}"
)
plt.legend([f"Trajectory A\n{legend_text}", "Trajectory B"])

plt.show()

# Plot LCSS matches
plt.figure(figsize=(10, 6))
plt.plot(x_A, y_A, "o-", label="Trajectory A")
plt.plot(x_B, y_B, "s-", label="Trajectory B")
for i, j in lcss_matches:
    plt.plot([x_A[i], x_B[j]], [y_A[i], y_B[j]], "k--")
plt.legend([f"Trajectory A\n{legend_text}", "Trajectory B"])
plt.title("LCSS Matches Between Trajectories")
plt.xlabel("X (Field Length)")
plt.ylabel("Y (Height)")
plt.grid(True)
plt.show()
