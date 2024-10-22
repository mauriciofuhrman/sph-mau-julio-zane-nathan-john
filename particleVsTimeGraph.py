import numpy as np
import matplotlib.pyplot as plt

# Data input (number of particles and execution time)
n_particles = np.array([8, 12, 36, 80, 490, 3380])
exec_time_parallel = np.array([2.03082, 2.05755, 2.90341, 4.08241, 15.3324, 127.358])
exec_time_serial = np.array([0.975238, 1.34032, 4.16485, 9.68295, 74.7066, 703.801])

# Create log-log plot using the data
plt.figure()

# Plot log-log data
# plt.loglog(n_particles, exec_time_serial, 'o-', label='Measured Time')
plt.loglog(n_particles, exec_time_serial, 'o-', label='Serial Execution Time')
plt.loglog(n_particles, exec_time_parallel, 's-', label='Parallel Execution Time')


# Labeling the plot
plt.xlabel('Number of Particles (n)')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time vs. Number of Particles (Log-Log Plot)')
plt.legend()
plt.grid(True, which="both", ls="--")

# Save the plot
plt.savefig('serial_parallel_particle_time_comparison.png')

# Show the plot
plt.show()
