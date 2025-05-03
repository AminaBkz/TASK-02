import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.uniform(-13, 13, 1000)

y =3*x + 8

noise = np.random.normal(scale=2, size=y.shape)
y_noisy = y + noise

np.savez("dataset.npz", x=x, y_noisy=y_noisy)

plt.figure(figsize=(8, 6))
plt.plot(x, y, color='blue', label='True Line: y = 3x + 8')
plt.scatter(x, y_noisy, color='red', s=10, alpha=0.5, label='Noisy Data')
plt.xlabel("x")
plt.ylabel("y'")

plt.title("True Model | Noisy Data")
plt.show()
