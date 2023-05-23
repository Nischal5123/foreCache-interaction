import numpy as np
import matplotlib.pyplot as plt

# Data
x = [0.1, 0.2, 0.3]
y = [[0.8, 0.5, 0.3], [0.7, 0.5, 0.3], [0.1, 0.5, 0.3]]
y_err = [[0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]

y_t = np.transpose(y) # Transpose y

# Create a new figure and set its size
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each data point with error bars
for i in range(len(x)):
    ax.errorbar(x[i], y_t[i], yerr=y_err[i], fmt='o', capsize=5, label='Data %d' % (i+1))

# Add a trend line
z = np.polyfit(x, np.mean(y_t, axis=1), 1)
p = np.poly1d(z)
ax.plot(x, p(x), 'r--', label='Trend line')

# Add labels and legends
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# Show the plot
plt.show()