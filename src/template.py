import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cupy as np
from time import time


start = time()
# Define the scale parameter (1/lambda), e.g., mean = 1.0
galaxy_radius = 3000*3.2615637769
galaxy_height = 250*3.2615637769

# Define the number of 3D points you want to generate
num_points = 100000000

# Generate cylindrical coordinates
p_radius = np.random.exponential(scale=galaxy_radius, size=num_points)
p_theta = np.random.random(num_points) * 2 *np.pi
p_z = np.random.exponential(scale=galaxy_height, size=num_points) * np.random.choice([-1,1], size = num_points)
points_cylindrical = np.vstack((p_radius,p_theta,p_z)).T


# Convert to Cartesian coordinates
p_x = p_radius * np.cos(p_theta)
p_y = p_radius * np.sin(p_theta)

cartesian_coords = np.vstack((p_x, p_y, p_z)).T

print(cartesian_coords)
print(time()-start)