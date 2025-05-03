import numpy as np

data = np.load("dataset.npz")
x = data["x"]
y_noisy = data["y_noisy"]

print(x, y_noisy)

# BELUGA APOLOGIES**2
