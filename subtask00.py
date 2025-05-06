import numpy as np
import matplotlib.pyplot as plt


x = np.random.uniform(-13, 13, 1000)

y_noisy = 3*x + 8 + np.random.normal(scale=3, size=1000)

plt.plot(x, 3*x + 8, 'b', label='a(x)')
plt.scatter(x, y_noisy, c='r', s=10, alpha=0.5, label='Noisy Data')
plt.xlabel("x")
plt.ylabel("y'")
plt.title("True Model | Noisy Data - subtask00")
plt.legend()
plt.show()


np.savez("dataset.npz", x=x, y_noisy=y_noisy)
