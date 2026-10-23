import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 100)

fig, ax = plt.subplots(figsize=(5, 2.7), layout="constrained")
ax.plot(x, x, label='лінія')
ax.plot(x, x**2, label="квадратна")
ax.plot(x, x**3, label="кубічна")

ax.set_xlabel('x label')
ax.set_xlabel('y label')
ax.set_title("Лінійні графіки")
ax.legend()
plt.show()