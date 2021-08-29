import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import CONFIG
# Set the carrying capacity:
K = [75, 1.5E2, 3E2]

# Extend the total time of the integration
delta_t = 0.1
N_0 = 1
r = 0.03
total_time = 500
time_range = np.arange(0, total_time, delta_t)
n_time_steps = int(total_time / delta_t)
# Set the storage vector so we don't rewrite our correct approach
N_t = np.zeros((len(K), n_time_steps))
N_t[:,0] = N_0

# Loop through each carrying capacity
for k in range(len(K)): 
    
    # Loop through each time step.
    for t in range(1, n_time_steps): 
    
        # Calculate the change in the number of cells. 
        dN = N_t[k, t - 1] * r * delta_t * (1 - N_t[k, t - 1] / K[k])
    
        # Update the number of cells at the current time point
        N_t[k, t] = N_t[k, t - 1] + dN


N_t = N_t.transpose()

time_range = time_range[:, np.newaxis] #TODO: fix this hardcoding

output_path = CONFIG.get_dataset_path_from_file(__file__)
print('dataset output path: {}'.format(output_path))
with open(output_path, 'wb') as f:
    print('dumping data...')
    print('shapes: {}, {}'.format(time_range.shape, N_t.shape))
    pickle.dump((time_range, N_t, K), f)
    
# Loop through the carrying capacities and plot every 100th point.
for i in range(len(K)):
    plt.plot(time_range[::100, 0], N_t[::100, i], '.', label='K = ' + str(K[i]))

    
# Add appropriate labels and legends. 
plt.xlabel('time  [min]')
plt.ylabel('number of cells')
plt.legend() 
plt.show()
