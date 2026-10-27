from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, make_regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

print("Привіт! Працюємо з множина регресією")

# Отримати набір даних із Каліфорнія житло - 1990
# as_frame - дані у вигляді DataFrmae
housing = fetch_california_housing(as_frame=True)

#Отримало сам DataFrame
df = housing.frame

# print(df.head()) # Вивести 5 перших рядків із даного dataset

# Обираємо ознаки для множиної регресії
df = df[[
    "MedInc", # Середній дохід населення району
    "HouseAge", # Середній вік будинку в районі
    "AveRooms", # Середня кількість кімнат в районі
    "MedHouseVal" # Середня ціна будинку - цільова ознака
]]

# Нова колонка - категоріальний поділ
df["Category"] = np.where(df["HouseAge"] > 30, "Old", "New")

print("Нова таблиця для множинної регресії")
# print(df.head())

# Виділяємо ознаки по X та Y
X = df.drop(columns="MedHouseVal")
# print(X.head())

y = df["MedHouseVal"]
# print(y.head())

# Розподіл даних і налаштовуємо random
# X - Список ознак по яких будемо робити аналіз - експертизу
# y - Значення, яке потрібно прогозувати - цільова ознака
# test_size - 20% це будуть дані для тесту
# 80% - це для навчання - виявлення закономірносте даних
# random_state=42 - для того, щоб дані завжди були однаковими
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42))

# проводимо обробку даних
numeric_features = [
    "MedInc",
    "HouseAge",
    "AveRooms",
]

categorical_features = [
    "Category"
]

preprocessor = ColumnTransformer(
    #Правила для обробки інформації
    transformers=[
        (
            # Назва трансформації
            "num",
            # Буде усі числові конки приводити до значення - близького до 1
            # Віднімаємо середнє і ділимо на стандартне відхилення
            StandardScaler(),
            # Це будемо робити на основі ось цих даних
            numeric_features,
        ),
        (
            "cat",
            # воно приводити дані до числоваого формату
            # при цьому буде лише 1 колонка у які буде вказано 0 або 1
            OneHotEncoder(drop="first"),
            categorical_features, # Застосовуємо до цих даних
        )
    ]
)

# Будуємо модель для роботи із ознаками
model = Pipeline([
    (
# назва 1 етапу
        "preprocessor",
# Попередня обробка даних: Стандартизація та Кодування Категоріальних ознак
        preprocessor
    ),
    (
        "regressor", # Назва 2 етапу
        LinearRegression() # Множина лінійна регресія
    )
])

# Тепер модель знає з якими даним її працювати
model.fit(X_train, y_train) # проходить машине навчання

# Отримаємо множину пронозів для тестових дани
y_pred = model.predict(X_test)

print(y_pred)

#Оцінка моделі
r2 = r2_score(y_test, y_pred) # порівняємо, що вийшло з тим, що малов вийти

# шукаємо абсолютну похибку
mea = mean_absolute_error(y_test, y_pred)

#Середньо квадратична похибка даних
rmse = root_mean_squared_error(y_test, y_pred)

print("r2 score:", r2)
print("mea score:", mea) # похибка очислень - 0.6 - якщо будинок коштує сотні тисяч - то похибка 60 тисяч
print("rmse score:", rmse) # Дана змінна реагує на досить великі відхилення від норми

# ==========================================
# 11. Коефіцієнти моделі
# ==========================================
# Формуємо список назв усіх ознак,
# які були використані моделлю після попередньої обробки.
feature_names = (
    # Додаємо числові ознаки.
    numeric_features +
    # Додаємо назви нових стовпців,
    # створених OneHotEncoder.
    list(
        # Отримуємо об'єкт попередньої обробки (ColumnTransformer)
        # із Pipeline.
        model.named_steps["preprocessor"]
                # Отримуємо трансформатор категоріальних ознак.
             .named_transformers_["cat"]
# Повертаємо назви стовпців після OneHotEncoder.
             # Наприклад:
             # Category → Category_New
             .get_feature_names_out(categorical_features)
    )
)
# Отримуємо коефіцієнти навченої моделі
# множинної лінійної регресії.
coef = model.named_steps["regressor"].coef_

# Створюємо DataFrame,
# у якому кожній ознаці відповідає її коефіцієнт.
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coef
})
# Виводимо заголовок.
print("\nКоефіцієнти моделі:")
# Виводимо таблицю коефіцієнтів.
print(coef_df)

# -- практична --
# завдання 1

california = fetch_california_housing(as_frame=True)

df = california.frame

features = ['MedInc', 'HouseAge', 'AveRooms']
X = df[features]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2_result = r2_score(y_test, y_pred)

results_table = pd.DataFrame({
    'Метрика': ['Коефіцієнт детермінації (R2)'],
    'Значення': [round(r2_result, 4)]
})

print(results_table)

# завдання 2

california = fetch_california_housing(as_frame=True)
df = california.frame

features = ['MedInc', 'HouseAge', 'AveRooms']
X = df[features]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_raw = LinearRegression()
model_raw.fit(X_train, y_train)
y_pred_raw = model_raw.predict(X_test)
r2_raw = r2_score(y_test, y_pred_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

comparison_table = pd.DataFrame({
    'Модель': ['linearRegression', 'linearRegression'],
    'Масштабування': ['Ні', 'StandardScaler'],
    'R2': [round(r2_raw, 4), round(r2_scaled, 4)]
})

print(comparison_table)

# завдання 3

california = fetch_california_housing(as_frame=True)
df = california.frame.copy()

numeric_features = ['MedInc', 'HouseAge', 'AveRooms']

np.random.seed(42)
mask = np.random.rand(len(df)) < 0.05
df.loc[mask, 'MedInc'] = np.nan

imputer = SimpleImputer(strategy='mean')
df[numeric_features] = imputer.fit_transform(df[numeric_features])

df['Category'] = np.where(df['HouseAge'] > 30, 'Old', 'New')

df_factorize = df.copy()
df_factorize['Category_encoded'] = pd.factorize(df_factorize['Category'])[0]

X_fact = df_factorize[numeric_features + ['Category_encoded']]
y_fact = df_factorize['MedHouseVal']

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fact, y_fact, test_size=0.2, random_state=42
)

model_fact = LinearRegression()
model_fact.fit(X_train_f, y_train_f)
r2_fact = r2_score(y_test_f, model_fact.predict(X_test_f))

ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe_encoded = ohe.fit_transform(df[['Category']])
ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(['Category']))

X_ohe = pd.concat([df[numeric_features].reset_index(drop=True), ohe_df], axis=1)
y_ohe = df['MedHouseVal'].reset_index(drop=True)

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_ohe, y_ohe, test_size=0.2, random_state=42
)

model_ohe = LinearRegression()
model_ohe.fit(X_train_o, y_train_o)
r2_ohe = r2_score(y_test_o, model_ohe.predict(X_test_o))

comparison_table = pd.DataFrame({
    'Модель': ['LinearRegression', 'LinearRegression'],
    'Категоріальне кодування': ['factorize', 'OneHotEncoder'],
    'R2': [round(r2_fact, 4), round(r2_ohe, 4)]
})

print(comparison_table)

# завдання 4

california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model_single = LinearRegression()
model_single.fit(X_train, y_train)
y_pred_single = model_single.predict(X_test)
r2_single = r2_score(y_test, y_pred_single)

comparison_table = pd.DataFrame({
    'Метод оцінки': ['Звичайне розбиття (Train / Test)', 'Крос-валідація (K-Fold, k=5)'],
    'Середній R2': [round(r2_single, 4), round(cv_mean, 4)],
    'Стандартне відхилення (std)': [np.nan, round(cv_std, 4)]
})

print(comparison_table)

r2_train = r2_score(y_train, model_single.predict(X_train))
print(f"\nR2 на навчальній вибірці (Train): {r2_train:.4f}")
print(f"R2 на тестовій вибірці (Test):   {r2_single:.4f}")

# -- домашня --
# завдання 1

X, y = make_regression(n_samples=2000, n_features=4, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

coefficients_table = pd.DataFrame({
    'Ознака': ['x1', 'x2', 'x3', 'x4'],
    'Коефіцієнт': [round(coef, 4) for coef in model.coef_]
})

print("Таблиця коефіцієнтів:")
print(coefficients_table)

r2_result = r2_score(y_test, y_pred)
metrics_table = pd.DataFrame({
    'Метрика': ['Метрика R2 (Test)'],
    'Значення': [round(r2_result, 4)]
})

print("Метрики моделі:")
print(metrics_table)

# завдання 2

X, y = make_regression(n_samples=2000, n_features=4, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_raw = LinearRegression()
model_raw.fit(X_train, y_train)
y_pred_raw = model_raw.predict(X_test)
r2_raw = r2_score(y_test, y_pred_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

comparison_table = pd.DataFrame({
    'Масштабування': ['Ні', 'StandardScaler'],
    'R2': [round(r2_raw, 4), round(r2_scaled, 4)]
})

print(comparison_table)

# завдання 3

X, y = make_regression(n_samples=2000, n_features=4, noise=10, random_state=42)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_raw = LinearRegression()
model_raw.fit(X_train_raw, y_train)
r2_raw = r2_score(y_test, model_raw.predict(X_test_raw))

X_missing = X.copy()
np.random.seed(42)
mask = np.random.rand(len(X_missing)) < 0.10
X_missing[mask, 0] = np.nan

X_train_miss, X_test_miss, _, _ = train_test_split(
    X_missing, y, test_size=0.2, random_state=42
)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_miss)
X_test_imputed = imputer.transform(X_test_miss)

model_imputed = LinearRegression()
model_imputed.fit(X_train_imputed, y_train)
r2_imputed = r2_score(y_test, model_imputed.predict(X_test_imputed))

comparison_table = pd.DataFrame({
    'Сценарій': ['Без пропусків', 'Після імпутації'],
    'R2': [round(r2_raw, 4), round(r2_imputed, 4)]
})

print(comparison_table)

# завдання 4

X, y = make_regression(n_samples=2000, n_features=4, noise=15, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

results_table = pd.DataFrame({
    'Метрика': ['Середній R2', 'Стандартне відх. (std)'],
    'Значення': [round(cv_mean, 4), round(cv_std, 4)]
})

print(results_table)