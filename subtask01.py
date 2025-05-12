import numpy as np
import matplotlib.pyplot as plt
import random

# Load the data (x and y values we need)
data = np.load("dataset.npz")
x = data["x"]
y = data["y_noisy"]

# Randomly guess the starting values for w and b
w = random.uniform(-1, 1)
b = random.uniform(-1, 1)

# Setting up gradient descent
lr = 0.01  # How fast we adjust w and b, I wanted to make it adjust dynamically, but too much hassle..
epochs = 1000  # Max number of updates
tolerance = 1e-5  # Stop when changes get too tiny
errors = []  # Track how bad our working is
prev_w, prev_b = w, b  # Store previous values to check progress

# Gradient Descent
for r in range(epochs):
    y_pred = w * x + b  # Our current y
    error = y_pred - y  # See how wrong we are
    mse = np.mean(error ** 2)  # Squared error so negatives don’t cancel out
    errors.append(mse)  # Keep track of errors

    # Update w and b based on how bad the predictions are
    w -= lr * np.mean(error * x) * 2
    b -= lr * np.mean(error) * 2

    # If we're barely changing anymore, just stop 'tolerance = 1r-5'
    if abs(w - prev_w) < tolerance and abs(b - prev_b) < tolerance:
        break

    prev_w, prev_b = w, b  # Save new values for the next check

# Print final results
print(f"Estimated line: y = {w:.3f}x + {b:.3f}")
print(f"Training stopped after {r} iterations")
print(f"Minimum error achieved: {min(errors):.13f}")

# ------> Plotting
plt.figure()
plt.scatter(x, y, color="#3e3ee5", alpha=0.5, label="Noisy Data")  # Messy data points
plt.plot(x, w * x + b, color='#08a60a', label=f"Estimated y={w:.2f}x+{b:.2f}")  # Our final trained line
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Approximation")
plt.legend()
plt.show()

# ------> How bad the error was over time (hopefully it keeps dropping)
plt.figure()
plt.plot(errors, color='#08a60a')
plt.xlabel("Progress")
plt.ylabel("Mean Squared Error")
plt.title("Error Reduction Over Time")
plt.show()
