import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the sine wave
plt.plot(x, y1, label='Sine Wave')

# Plot the cosine wave
plt.plot(x, y2, label='Cosine Wave')

# Plot the tangent wave
plt.plot(x, y3, label='Tangent Wave')

# Adding labels and title
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Sine, Cosine, and Tangent Waves')

# Adding a legend
plt.legend()

# Setting y limits to avoid extreme values for tangent
plt.ylim(-10, 10)

# Display the plot
plt.show()
