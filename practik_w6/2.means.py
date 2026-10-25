import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

grades = np.array([
    72, 75, 80, 82, 78,
    75, 74, 81, 79, 76,
    75, 83, 77, 84, 80,
    75, 73, 85, 79, 82,
    75, 78, 81, 76, 80,
    75, 74, 83, 79, 77
])

# хочемо зломати систему додамо випадково дуже високу оцінку
grades = np.append(grades, 1000)

#Статистичні обчислюємо
mean_grade = np.mean(grades)
median_grade = np.median(grades)
mode_grade = stats.mode(grades, keepdims=True).mode[0] #Мода

fig, ax = plt.subplots(figsize=(11,5))
#Кількість стовпчиків у графіку
ax.hist(grades, bins=15,
        color="skyblue", edgecolor="black")

ax.axvline(mean_grade, color="red",
           linewidth=3,
           label=f"Середнє = {mean_grade:.1f}")

ax.axvline(median_grade, color="green",
           linewidth=3,
           label=f"Медіана = {median_grade:.1f}")

ax.axvline(mode_grade, color="blue",
           linewidth=3,
           linestyle="--",
           label=f"Мода = {mode_grade:.1f}")

ax.set_title("Результати контрольної роботи")
ax.set_xlabel("Бали")
ax.set_ylabel("Кількість студентів")

ax.legend()

plt.tight_layout()
plt.show()

print("Оцінки:")
print(grades)

print(f"\nСереднє : {mean_grade:.2f}")
print(f"\nМедіана : {median_grade:.2f}")
print(f"\nМода : {mode_grade:.2f}")