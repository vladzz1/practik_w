import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#Попит на морозиво
temperature = np.array([2, 5, 8, 12, 15, 18, 20, 22, 25, 28, 30, 33])
ice_cream_sales = np.array([15, 22, 35, 60, 90, 130, 160, 20, 230, 250, 243, 210])

print("Температура:", temperature)
print("Кількість порцій морозива:", ice_cream_sales)

X = temperature.reshape(-1, 1) # 2D масив ознак
y = ice_cream_sales
# print("X", X)

#Будуємо модель для регресії полінома і даємо степінь
model = make_pipeline(PolynomialFeatures(4), LinearRegression())
model.fit(X, y) # передаємо значення у модель для навчання

x_smooth = np.linspace(temperature.min(), temperature.max(), 200).reshape(-1, 1)
y_smooth = model.predict(x_smooth) # Будемо робити прогнозування ціни для наших даних

new_temp = np.array([[27]])
predicted = model.predict(new_temp) # Прогноз для температури в 27 градусів

print(f"Прогноз для температури 27 C: {predicted[0]:.0f} шт")
print(f"R^2 моделі = {model.score(X, y):.3f}")

plt.scatter(temperature,ice_cream_sales, color="orange", label = "Реальні продажі")
plt.plot(x_smooth,y_smooth, color="navy", label = "Понілом 2-го ступеня")
plt.xlabel("Температура")
plt.ylabel("Продажі шт")
plt.legend()
plt.show()

# -- практична --
# завдання 1

X, y = make_regression(
    n_samples=200, n_features=1, noise=15, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

X_grid = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_grid_poly = poly.transform(X_grid)

y_pred_lin = lin_reg.predict(X_grid)
y_pred_poly = poly_reg.predict(X_grid_poly)

plt.figure(figsize=(10, 6))
plt.scatter(
    X, y, color='blue', alpha=0.6, label='Вихідні точки (make_regression)'
)
plt.plot(X_grid, y_pred_lin, color='red', linewidth=2, label='Лінійна регресія (degree=1)')
plt.plot(
    X_grid,
    y_pred_poly,
    color='green',
    linewidth=2,
    linestyle='--',
    label='Поліноміальна регресія (degree=2)',
)

plt.title('Порівняння лінійної та поліноміальної регресії')
plt.xlabel('Ознака X')
plt.ylabel('Цільова змінна y')
plt.legend()
plt.grid(True)
plt.show()

# завдання 2

X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = [2, 5, 10]
data_records = []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    r2_train = r2_score(y_train, model.predict(X_train_poly))
    r2_test = r2_score(y_test, model.predict(X_test_poly))

    data_records.append(
        {
            'Ступінь полінома': deg,
            'R2 поїзд': f'{r2_train:.4f}',
            'Тест R2': f'{r2_test:.4f}',
        }
    )

df = pd.DataFrame(data_records)
print(df)

# завдання 3

X, y = make_regression(
    n_samples=500, n_features=20, n_informative=5, noise=10, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
r2_lr = r2_score(y_test, lr.predict(X_test))

lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)
r2_lasso = r2_score(y_test, lasso.predict(X_test))

zeroed_coefficients = np.sum(np.isclose(lasso.coef_, 0, atol=1e-5))

print(f'R2 LinearRegression на тесті: {r2_lr:.4f}')
print(f'R2 Lasso (alpha=0.1) на тесті: {r2_lasso:.4f}')
print(f'Кількість коефіцієнтів, які Lasso занулила: {zeroed_coefficients}')