import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -- практична --
# завдання 1

diabetes = load_diabetes()

X_full = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_full = diabetes.target

X_single = diabetes.data[:, [2]]

print(f"Назва обраної ознаки: {diabetes.feature_names[2].upper()} (Індекс маси тіла)")
print(f"Розмірність матриці ознаки: {X_single.shape}")
print(f"Розмірність вектора цільової змінної: {y_full.shape}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X_single, 
    y_full, 
    test_size=0.20, 
    random_state=42
)

print(f"Розмір навчальної вибірки (X_train): {X_train.shape}")
print(f"Розмір тестової вибірки (X_test):  {X_test.shape}\n")

model = LinearRegression()
model.fit(X_train, y_train)

print("Модель успішно навчено!")
print(f"Коефіцієнт нахилу (w1 / slope):   {model.coef_[0]:.4f}")
print(f"Вільний член (w0 / intercept):  {model.intercept_:.4f}\n")

y_pred = model.predict(X_test)

comparison_df = pd.DataFrame({
    'Реальне значення (y_test)': y_test[:10],
    'Передбачене значення (y_pred)': np.round(y_pred[:10], 1),
    'Абсолютна помилка (Різниця)': np.round(np.abs(y_test[:10] - y_pred[:10]), 1)
})

print("--- Порівняння перших 10 результатів на тестовій вибірці ---")
print(comparison_df.to_string(index=False))

# завдання 2

plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, color='dodgerblue', alpha=0.7, edgecolors='k', s=50, label='Реальні дані (Test)')

plt.plot(X_test, y_pred, color='crimson', linewidth=3, label='Пряма лінійної регресії')

plt.title("Візуалізація моделі лінійної регресії (Diabetes Dataset)", fontsize=14, fontweight='bold')
plt.xlabel("Індекс маси тіла (BMI, стандартизований)", fontsize=12)
plt.ylabel("Прогресування хвороби (Target)", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.show()

# завдання 3

mae_val = mean_absolute_error(y_test, y_pred)
mse_val = mean_squared_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)

metrics_table = pd.DataFrame({
    'Метрика': ['MAE', 'MSE', 'R2'],
    'Значення': [round(mae_val, 4), round(mse_val, 4), round(r2_val, 4)]
})

print("--- Результати оцінки якості моделі ---")
print(metrics_table.to_string(index=False))

# завдання 4

def gradient_descent(X, y, learning_rate=0.1, epochs=5000):
    """
    X: одновимірний або двовимірний масив ознак
    y: вектор цільової змінної
    """
    X = X.flatten()
    n = len(y)

    w = 0.0
    b = 0.0

    for _ in range(epochs):
        y_pred_current = w * X + b
        
        dw = (-2 / n) * np.sum(X * (y - y_pred_current))
        db = (-2 / n) * np.sum(y - y_pred_current)
        
        w -= learning_rate * dw
        b -= learning_rate * db
        
    return w, b

w_custom, b_custom = gradient_descent(X_train, y_train, learning_rate=0.2, epochs=10000)

print("--- Коефіцієнти моделей ---")
print(f"Custom GD:         w = {w_custom:.4f}, b = {b_custom:.4f}")
print(f"LinearRegression:  w = {model.coef_[0]:.4f}, b = {model.intercept_:.4f}\n")

X_test_flat = X_test.flatten()
y_pred_custom = w_custom * X_test_flat + b_custom

mae_custom = mean_absolute_error(y_test, y_pred_custom)
mae_sklearn = mean_absolute_error(y_test, y_pred)

comparison_table = pd.DataFrame({
    'Модель': ['Власний Градієнтний Спуск (GD)', 'Sklearn LinearRegression'],
    'Коефіцієнт w1 (Slope)': [round(w_custom, 2), round(model.coef_[0], 2)],
    'Вільний член w0 (Intercept)': [round(b_custom, 2), round(model.intercept_, 2)],
    'Тестовий MAE': [round(mae_custom, 4), round(mae_sklearn, 4)]
})

print("--- Порівняння результатів ---")
print(comparison_table.to_string(index=False))

# -- домашня --
# завдання 1

housing = fetch_california_housing(as_frame=True)
df_full = housing.frame

df_sampled = df_full.sample(n=5000, random_state=42)

X_single = df_sampled[['MedInc']]
y_target = df_sampled['MedHouseVal']

print(f"Розмірність вибірки після фільтрації: {X_single.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_single, 
    y_target, 
    test_size=0.20, 
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2_val = r2_score(y_test, y_pred)

result_table = pd.DataFrame({
    'Модель': ['LinearRegression'],
    'Ознака': ['MedInc'],
    'R²': [round(r2_val, 4)]
})

print("\n--- Фінальна таблиця результатів ---")
print(result_table.to_string(index=False))

# завдання 2

y_target = df_sampled['MedHouseVal']

X_1 = df_sampled[['MedInc']]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_1, y_target, test_size=0.20, random_state=42)

model1 = LinearRegression()
model1.fit(X_train1, y_train1)
y_pred1 = model1.predict(X_test1)
r2_model1 = r2_score(y_test1, y_pred1)

X_2 = df_sampled[['MedInc', 'AveRooms']]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_2, y_target, test_size=0.20, random_state=42)

model2 = LinearRegression()
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)
r2_model2 = r2_score(y_test2, y_pred2)

comparison_table = pd.DataFrame({
    'Модель': ['LinearRegression', 'LinearRegression'],
    'Ознаки': ['MedInc', 'MedInc, AveRooms'],
    'R²': [round(r2_model1, 4), round(r2_model2, 4)]
})

print("--- Порівняння якості моделей (завдання 2) ---")
print(comparison_table.to_string(index=False))

# завдання 3

mae_val = mean_absolute_error(y_test2, y_pred2)
mse_val = mean_squared_error(y_test2, y_pred2)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_test2, y_pred2)

metrics_table = pd.DataFrame({
    'Метрика': ['MAE', 'MSE', 'RMSE', 'R²'],
    'Значення': [round(mae_val, 4), round(mse_val, 4), round(rmse_val, 4), round(r2_val, 4)]
})

print("--- Результати оцінки якості моделі (MedInc, AveRooms) ---")
print(metrics_table.to_string(index=False))

# завдання 4

plt.figure(figsize=(9, 7))

plt.scatter(y_test2, y_pred2, color='mediumpurple', alpha=0.6, edgecolors='k', s=40, label='Реальні vs Передбачені')

min_val = min(y_test2.min(), y_pred2.min())
max_val = max(y_test2.max(), y_pred2.max())
line_coords = np.linspace(min_val, max_val, 100)

plt.plot(line_coords, line_coords, color='darkorange', linestyle='--', linewidth=2.5, label='Ідеальний прогноз (y = x)')

plt.title("Порівняння реальних і передбачених значень вартості нерухомості", fontsize=13, fontweight='bold')
plt.xlabel("Реальні значення (y_test)", fontsize=11)
plt.ylabel("Передбачені моделлю значення (y_pred)", fontsize=11)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=11)

plt.tight_layout()

output_filename = 'actual_vs_predicted_california.png'
plt.savefig(output_filename, dpi=300)
print(f"Графік успішно побудовано та збережено у файл: '{output_filename}'")

plt.show()