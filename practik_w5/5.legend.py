import matplotlib.pyplot as plt  # Імпортуємо модуль pyplot для побудови графіків
import numpy as np               # Імпортуємо бібліотеку NumPy для роботи з масивами

data1 = np.array([1, 2, 3, 4, 5, 6])
data2 = np.array([2, 3, 2, 5, 4, 7])
data3 = np.array([1, 4, 2, 6, 3, 5])

# Створюємо фігуру (Figure) та область побудови графіка (Axes)
# figsize=(5, 2.7) задає розмір фігури у дюймах
fig, ax = plt.subplots(figsize=(5, 2.7))

# Будуємо перший графік.
# np.arange(len(data1)) створює значення осі X від 0 до кількості елементів data1 - 1.
# label задає назву графіка для легенди.
ax.plot(np.arange(len(data1)), data1, label='Дані 1')

# Будуємо другий графік
ax.plot(np.arange(len(data2)), data2, label='Дані 2')

# Будуємо третій графік.
# 'd' означає, що точки будуть відображатися у вигляді ромбів (diamond)
ax.plot(np.arange(len(data3)), data3, 'd', label='Дані 3')

# Додаємо підпис осі X
ax.set_xlabel('Номер елемента')

# Додаємо підпис осі Y
ax.set_ylabel('Значення')

# Додаємо заголовок графіка
ax.set_title('Порівняння трьох наборів даних')

# Додаємо легенду, яка показує назви всіх графіків
ax.legend()

# Відображаємо сітку
ax.grid(True)

# Показуємо графік
plt.show()