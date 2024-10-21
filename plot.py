import matplotlib.pyplot as plt

num_cores = [2, 4, 8, 16, 32, 64]
times = [543.895, 294.27, 160.358, 109.14, 121.182, 129.396]
# times = [107.118, 124.73, 162.398, 292.55, 545.153] # num_particles = 3380
# 157.999

plt.plot(num_cores, times, 'o-', linewidth=2, label='Execution Time')
plt.xlabel('Number of Cores')
plt.ylabel('Time (seconds)')
plt.title('Scalability Plot')
plt.grid(True)
plt.legend()

plt.savefig('scalability_plot.png')