import matplotlib.pyplot as plt
import numpy as np
import random

print("-- Працюємо із графіками --")

# набір величин
x = np.linspace(0, 2 * np.pi, 200)
# знаходимо sin від x
y = np.sin(x)
#Створюємо фігуру для малювання
fig, ax = plt.subplots() # Figure - усе вікно графіку
ax.plot(x, y) # - звичайний лінійний графік
plt.show()

# -- практична --
# завдання 1

# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
# fig, ax = plt.subplots()
# ax.plot(x, y, 'r-o')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title("Графік функції y = x³")
# plt.show()

# завдання 2

# x = [1, 2, 3, 4, 5, 6]
# y = [150, 200, 250, 300, 280, 320]
# fig, ax = plt.subplots()
# ax.bar(x, y, color='blue')
# ax.set_xlabel('X Місяці')
# ax.set_ylabel('Y Продажі')
# ax.set_title('Графік продажів по місяцях')
# plt.show()

# завдання 3

# x = ["Продукти", "Одяг", "Техніка"]
# y = [40, 30, 30]
# fig, ax = plt.subplots()
# ax.pie(y, labels=x, autopct='%1.1f%%')
# ax.set_title('Розподіл продажів між категоріями')
# plt.show()

# завдання 4

# x = [160, 165, 170, 175, 180]
# y = [55, 60, 70, 72, 80]
# fig, ax = plt.subplots()
# ax.scatter(x, y, color='green')
# ax.set_xlabel('X Зріст (см)')
# ax.set_ylabel('Y Вага (кг)')
# ax.set_title('Залежність ваги від зросту')
# ax.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# завдання 5

# x = [0, 4, 8, 12, 16, 20, 24]
# y = [15, 14, 18, 22, 25, 20, 17]
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_xlabel('X години')
# ax.set_ylabel('Y температура')
# ax.set_title("Температура протягом доби")
# ax.grid(True)
# plt.show()

# завдання 6

# x = ["Alex", "Irene", "John", "Mary", "Boris"]
# y = [85, 92, 78, 90, 88]
# fig, ax = plt.subplots()
# ax.bar(x, y, color='orange')
# ax.set_xlabel('X імена студентів')
# ax.set_ylabel('Y їхні результати')
# ax.set_title('Результати тесту')
# plt.show()

# -- практична --
# завдання 1

# months = ["Січень", "Лютий", "Березень", "Квітень"]
# plan = [100, 120, 150, 130]
# fact = [95, 130, 145, 150]
# fig, ax = plt.subplots()
# ax.plot(months, plan, 'b-o', label='План')
# ax.plot(months, fact, 'r-s', label='Факт')
# ax.set_xlabel('X Місяці')
# ax.set_ylabel('Y Обсяг продажів')
# ax.set_title('Порівняння планових та фактичних продажів')
# ax.legend()
# ax.grid(True, linestyle='--', alpha=0.5)
# plt.show()

# завдання 2

# ages = [random.randint(10, 80) for _ in range(100)]
# mean_age = sum(ages) / len(ages)
# fig, ax = plt.subplots()
# ax.hist(ages, bins=10, color='skyblue', edgecolor='black')
# ax.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Середній вік: {mean_age:.1f}')
# ax.set_xlabel('X Вік')
# ax.set_ylabel('Y Кількість людей')
# ax.set_title('Розподіл вікових груп серед 100 осіб')
# ax.legend()
# plt.show()

# завдання 3

# group_A = [random.randint(55, 100) for _ in range(30)]
# group_B = [random.randint(50, 95) for _ in range(30)]
# group_C = [random.randint(65, 100) for _ in range(30)]
# data = [group_A, group_B, group_C]
# fig, ax = plt.subplots()
# ax.boxplot(data, tick_labels=['Група А', 'Група Б', 'Група В'])
# ax.set_xlabel('X Групи студентів')
# ax.set_ylabel('Y Оцінки за іспит')
# ax.set_title('Порівняння розкиду оцінок між групами')
# ax.yaxis.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# завдання 4

# dates = ["12.10", "13.10", "14.10", "15.10", "16.10", "17.10", "18.10"]
# temperature = [15, 12, 13, 9, 10, 11, 15]
# humidity = [54, 46, 63, 72, 74, 74, 72]
# fig, ax1 = plt.subplots()
# line1 = ax1.plot(dates, temperature, 'r-o', label='Температура (°C)')
# ax1.set_xlabel('Дати за тиждень')
# ax1.set_ylabel('Температура (°C)', color='red')
# ax1.tick_params(axis='y', labelcolor='red')
# ax2 = ax1.twinx()
# line2 = ax2.plot(dates, humidity, 'b-s', label='Вологість (%)')
# ax2.set_ylabel('Вологість (%)', color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')
# ax1.set_title('Метеодані за тиждень: Температура та Вологість')
# lines = line1 + line2
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper left')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# завдання 5

# hours = list(range(24))
# server_load = [random.randint(15, 95) for _ in range(24)]
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(hours, server_load, color='purple', linewidth=2, label='Навантаження')
# ax.fill_between(hours, server_load, color='purple', alpha=0.3)
# ax.grid(True, linestyle='--', alpha=0.6)
# ax.set_xlabel('Година доби')
# ax.set_ylabel('Навантаження сервера (%)')
# ax.set_title('Погодинне навантаження сервера за добу')
# ax.set_xticks(hours)
# ax.legend()
# plt.show()

# завдання 6

# months = ["Січ", "Лют", "Бер", "Квіт", "Трав", "Черв"]
# conversion = [random.randint(2, 7) for _ in range(6)]
# retention = [random.randint(30, 60) for _ in range(6)]
# avg_check = [random.randint(400, 800) for _ in range(6)] 
# orders = [random.randint(100, 500) for _ in range(6)]
# fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# 
# axs[0, 0].plot(months, conversion, color='blue', marker='o')
# axs[0, 0].set_title('Конверсія (%)')
# axs[0, 0].grid(True, linestyle='--')
# 
# axs[0, 1].plot(months, retention, color='green', marker='s')
# axs[0, 1].set_title('Утримання користувачів (%)')
# axs[0, 1].grid(True, linestyle='--')
# 
# axs[1, 0].bar(months, avg_check, color='orange')
# axs[1, 0].set_title('Середній чек (грн)')
# axs[1, 0].grid(True, linestyle='--')
# 
# axs[1, 1].plot(months, orders, color='purple', marker='^')
# axs[1, 1].set_title('Кількість замовлень (шт)')
# axs[1, 1].grid(True, linestyle='--')
# 
# fig.suptitle('Аналітика ключових метрик продукту за півріччя', fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()