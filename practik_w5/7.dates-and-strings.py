import matplotlib.pyplot as plt                    # Імпортуємо модуль pyplot для побудови графіків
import numpy as np                                 # Імпортуємо бібліотеку NumPy
from matplotlib.dates import ConciseDateFormatter  # Імпортуємо форматувач дат

# Фіксуємо генератор випадкових чисел,
# щоб при кожному запуску отримувати однаковий результат
np.random.seed(42)

# Створюємо фігуру та область побудови графіка
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

# Створюємо масив дат від 15.11.2021 до 25.12.2021
# Крок між сусідніми датами становить 1 годину
dates = np.arange(
    np.datetime64('2021-11-15'),
    np.datetime64('2021-12-25'),
    np.timedelta64(1, 'h')
)

# Створюємо випадкові дані.
# np.random.randn() генерує випадкові числа,
# а np.cumsum() накопичує їх суму,
# утворюючи плавний графік, схожий на зміну курсу акцій
data = np.cumsum(np.random.randn(len(dates)))

# Будуємо графік
ax.plot(dates, data)

# Додаємо заголовок
ax.set_title("Зміна значень у часі")

# Додаємо підписи осей
ax.set_xlabel("Дата")
ax.set_ylabel("Значення")

# Форматуємо підписи дат так,
# щоб вони були короткими та читабельними
ax.xaxis.set_major_formatter(
    ConciseDateFormatter(ax.xaxis.get_major_locator())
)

# Відображаємо сітку
ax.grid(True)

# Відображаємо графік
plt.show()