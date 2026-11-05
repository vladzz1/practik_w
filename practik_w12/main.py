from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, make_regression, load_diabetes
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn # Модулі для створення нейронної мережі
import torch.optim as optim # Оптимізатори для навчання мережі

# Для робота dataset
iris = load_iris()

# print("Iris data loaded", iris.data.shape)
# print("Iris data loaded", iris.data[:5])
X = iris.data

y = iris.target
# print(X.shape, y.shape)
#print("y", y)
print("Характеристики")
for feature in iris.feature_names:
    print(" - ", feature)

print("\nКласи:")
for class_name in iris.target_names:
    print(" - ", class_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print("Навчальні приклади:", len(X_train))
# print("Тестові приклади:", len(X_test))

# print("X_train:", X_train[:5])
# print("X_test:", X_test[:5])

#Маштабування даних
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print("X_train:", X_train[:5])
# print("X_test:", X_test[:5])

#Перетворюємо дані в тензор
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Робимо багатошаровий персептрон MLP
class IrisMLP(nn.Module):
    def __init__(self):
        super(IrisMLP, self).__init__()

        #Стоврюємо шари для мережі
        self.model = nn.Sequential(
            # із 4 ознаки робимо 32 ознаки
            nn.Linear(4, 32),

            nn.ReLU(),

            nn.Linear(32, 8),
            nn.ReLU(),

            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.model(x)

model = IrisMLP()
print("Архітектура MLP")
print(model)

# Будуємо функцію втрат

creterion = nn.CrossEntropyLoss()
# Налаштовуємо оптимізатор
# Використовується для зміни Ваги нейронної мережі

optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100

loss_history = []

acc_history = []

for epoch in range(epochs):
    model.train()

    outputs = model(X_train_tensor)

    loss = creterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


    predictions = torch.argmax(outputs, 1)


    corect = (predictions == y_train_tensor)
    acc = corect.float().mean()

    loss_history.append(loss.item())
    acc_history.append(acc.item())

    if epoch % 20 == 0:
        print(f" - [{epoch+1:3}/{epochs}]")
        print(f" - Loss: {loss.item():.4f}")
        print(f" - Accuracy: {acc.item()*100:.2f}")

model.eval()

# Відключаємо обчислення градієнтів
with torch.no_grad():

    # Передаємо тестові дані через мережу - Передати тестові дані через навчену нейронну мережу та отримати її прогнози.
    test_outputs = model(X_test_tensor)


    # Вибираємо клас з найбільшим значенням
    test_predictions = torch.argmax(test_outputs, dim=1)

# Перетворюємо Tensor у NumPy
predictions_numpy = test_predictions.numpy()


# Обчислюємо Accuracy
test_accuracy = accuracy_score(y_test, predictions_numpy)

# Виводимо результат
print("\n================================")

# Виводимо точність
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("================================")

# Створюємо звіт класифікації
report = classification_report(
    y_test,                          # Правильні класи
    predictions_numpy,               # Передбачені класи
    target_names=iris.target_names   # Назви класів
)


# Виводимо звіт
print("\nClassification Report:")
# precision - Серед усіх квіток, які модель назвала певним класом, скільки дійсно належать до цього класу?
# recall - Скільки реальних об'єктів певного класу модель змогла знайти?
# F1-score — це показник, який одночасно враховує Precision та Recall.
print(report)

# Обчислюємо матрицю помилок - покаже, які класи шутає між собою модель
cm = confusion_matrix(y_test, predictions_numpy)


# Виводимо матрицю в консоль
print("\nConfusion Matrix:")
print(cm)

# Візуалізація матриці
# Створюємо область для графіка
plt.figure(figsize=(7, 6))


# Відображаємо матрицю як зображення
plt.imshow(cm, interpolation="nearest")


# Додаємо заголовок
plt.title("Confusion Matrix — Iris MLP")


# Додаємо назву осі X
plt.xlabel(
    "Передбачений клас"
)


# Додаємо назву осі Y
plt.ylabel(
    "Справжній клас"
)


# Створюємо список позицій класів
classes = [
    "setosa",
    "versicolor",
    "virginica"
]


# Встановлюємо підписи осі X
plt.xticks(
    range(3),
    classes
)


# Встановлюємо підписи осі Y
plt.yticks(
    range(3),
    classes
)


# Додаємо числові значення всередину матриці
for i in range(3):
    # Перебираємо стовпці
    for j in range(3):
        # Виводимо значення конкретної клітинки
        plt.text(
            j,                    # Позиція X
            i,                    # Позиція Y
            cm[i, j],             # Значення
            ha="center",          # Горизонтальне вирівнювання
            va="center"           # Вертикальне вирівнювання
        )
# Автоматично налаштовуємо відступи
plt.tight_layout()


# Показуємо графік
plt.show()

# Створюємо новий графік
plt.figure(
    figsize=(8, 5)
)


# Малюємо зміну Loss
plt.plot(
    loss_history
)


# Назва осі X
plt.xlabel(
    "Epoch"
)


# Назва осі Y
plt.ylabel(
    "Loss"
)


# Заголовок графіка
plt.title(
    "Зміна помилки під час навчання"
)


# Вмикаємо сітку
plt.grid()


# Показуємо графік
plt.show()

# Створюємо новий графік
plt.figure(
    figsize=(8, 5)
)


# Малюємо зміну Accuracy
plt.plot(
    acc_history
)


# Назва осі X
plt.xlabel(
    "Epoch"
)


# Назва осі Y
plt.ylabel(
    "Accuracy"
)


# Заголовок
plt.title(
    "Точність MLP під час навчання"
)


# Вмикаємо сітку
plt.grid()


# Показуємо графік
plt.show()


# Створюємо характеристики нової квітки
sample = [[
    5.1,                              # Довжина чашолистка
    3.5,                              # Ширина чашолистка
    1.4,                              # Довжина пелюстки
    0.2                               # Ширина пелюстки
]]


# Масштабуємо нові дані
# використовуючи scaler, навчений на TRAIN
sample_scaled = scaler.transform(
    sample
)


# Перетворюємо нові дані у Tensor
sample_tensor = torch.tensor(

    sample_scaled,

    dtype=torch.float32
)


# Переводимо модель у режим оцінювання
model.eval()


# Відключаємо градієнти
with torch.no_grad():

    # Передаємо нову квітку в MLP
    output = model(
        sample_tensor
    )


    # Знаходимо індекс найбільш імовірного класу
    prediction = torch.argmax(
        output,
        dim=1
    ).item()

# Виводимо роздільник
print("\n================================")


# Виводимо характеристики квітки
print("Характеристики нової квітки:")
print(sample)

# Виводимо передбачений клас
print("\nПередбачений клас:")

# Отримуємо назву класу за його індексом
print(
    iris.target_names[prediction]
)

# Виводимо роздільник
print("================================")

# -- практична --
# завдання 1

X, y = make_regression(n_samples=2000, n_features=10, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

activations = {
    'ReLU': nn.ReLU(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU(),
    'Swish': nn.SiLU()
}

print("=" * 34)
print(f"{'Активація':<12} | {'MAE':<8} | {'R2':<6}")
print("-" * 34)
loss_history_base = []

for name, act_fn in activations.items():
    model = nn.Sequential(
        nn.Linear(10, 32),
        act_fn,
        nn.Linear(32, 1)
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if name == 'ReLU':
            loss_history_base.append(loss.item())
        
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).numpy()
        
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"{name:<12} | {mae:<8.4f} | {r2:<6.4f}")

# завдання 2

print("=" * 60)
print(f"{'Активація':<12} | {'MAE (Без)':<9} | {'MAE (З DO)':<10} | {'R2 (Без)':<8} | {'R2 (З DO)':<8}")
print("-" * 60)
loss_history_dropout = []

for name, act_fn in activations.items():
    
    model_dropout = nn.Sequential(
        nn.Linear(10, 32),
        act_fn,
        nn.Dropout(p=0.3),
        nn.Linear(32, 1)
    )
    
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        if name == 'ReLU':
            loss_history_dropout.append(loss.item())
            
    model_dropout.eval()
    
    with torch.no_grad():
        pred_do = model_dropout(X_test_t).numpy()
    
    mae_do = mean_absolute_error(y_test, pred_do)
    r2_do = r2_score(y_test, pred_do)
    
    print(f"{name:<12} | {mae:<9.4f} | {mae_do:<10.4f} | {r2:<8.4f} | {r2_do:<8.4f}")

# завдання 3

loss_history_bn = []

model_bn = nn.Sequential(
    nn.Linear(10, 32),
    nn.BatchNorm1d(32),
    act_fn,
    nn.Linear(32, 1)
)

optimizer_bn = optim.Adam(model_bn.parameters(), lr=0.01)
for epoch in range(500):
    model_bn.train()
    optimizer_bn.zero_grad()
    loss = criterion(model_bn(X_train_t), y_train_t)
    loss.backward()
    optimizer_bn.step()
    loss_history_bn.append(loss.item())

model_bn.eval()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 501), loss_history_base, label='Без BatchNorm (Base)', color='orange', alpha=0.8)
plt.plot(range(1, 501), loss_history_dropout, label='З Dropout', color='red', alpha=0.8)
plt.plot(range(1, 501), loss_history_bn, label='З BatchNorm1d', color='blue', linestyle='--', alpha=0.8)

plt.title('Вплив Batch Normalization на динаміку Loss')
plt.xlabel('Епоха навчання')
plt.ylabel('Loss (MSE)')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()

# завдання 4

model_both = nn.Sequential(
    nn.Linear(10, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(32, 1)
)

for epoch in range(500):
    model_both.train()
    optimizer_bn.zero_grad()
    loss = criterion(model_both(X_train_t), y_train_t)
    loss.backward()
    optimizer_bn.step()

model_both.eval()

with torch.no_grad():
    pred_base = model(X_test_t).numpy()
    pred_do = model_dropout(X_test_t).numpy()
    pred_bn = model_bn(X_test_t).numpy()
    pred_both = model_both(X_test_t).numpy()

results = {
    "ReLU": (mean_absolute_error(y_test, pred_base), r2_score(y_test, pred_base)),
    "ReLU + Dropout": (mean_absolute_error(y_test, pred_do), r2_score(y_test, pred_do)),
    "ReLU + BatchNorm": (mean_absolute_error(y_test, pred_bn), r2_score(y_test, pred_bn)),
    "ReLU + Dropout + BatchNorm": (mean_absolute_error(y_test, pred_both), r2_score(y_test, pred_both))
}

print("=" * 47)
print(f"{'Конфігурація':<26} | {'MAE':<8} | {'R2':<6}")
print("-" * 47)
for name, (mae, r2) in results.items():
    print(f"{name:<26} | {mae:<8.4f} | {r2:<6.4f}")

# -- домашня --
# завдання 1

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

print("\n--- Розміри матриць (PyTorch Tensors) ---")
print(f"X_train_t shape: {X_train_t.shape}")
print(f"y_train_t shape: {y_train_t.shape}")
print(f"X_test_t shape:  {X_test_t.shape}")
print(f"y_test_t shape:  {y_test_t.shape}\n")

# завдання 2

n_features = X_train_t.shape[1]

model_diabetes = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.ReLU(),
    
    nn.Linear(64, 32),
    nn.ReLU(),
    
    nn.Linear(32, 16),
    nn.ReLU(),
    
    nn.Linear(16, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model_diabetes.parameters(), lr=0.001)

loss_history_diabetes = []

for epoch in range(50):
    model_diabetes.train()
    optimizer.zero_grad()
    
    outputs = model_diabetes(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    loss.backward()
    optimizer.step()
    
    loss_history_diabetes.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Епоха [{epoch+1}/50], Loss (MSE): {loss.item():.4f}")

# завдання 3

model_l2 = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.ReLU(),
    
    nn.Linear(64, 32),
    nn.ReLU(),
    
    nn.Linear(32, 16),
    nn.ReLU(),
    
    nn.Linear(16, 1)
)

criterion = nn.MSELoss()

optimizer_l2 = optim.Adam(model_l2.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(50):
    model_l2.train()
    optimizer_l2.zero_grad()
    loss = criterion(model_l2(X_train_t), y_train_t)
    loss.backward()
    optimizer_l2.step()

model_diabetes.eval()
model_l2.eval()

with torch.no_grad():
    pred_base = model_diabetes(X_test_t).numpy()
    pred_l2 = model_l2(X_test_t).numpy()

mae_base = mean_absolute_error(y_test, pred_base)
r2_base = r2_score(y_test, pred_base)

mae_l2 = mean_absolute_error(y_test, pred_l2)
r2_l2 = r2_score(y_test, pred_l2)

print(f"{'\nКонфігурація':<23} | {'MAE':<8} | {'R2':<6}")
print("-" * 44)
print(f"{'Без регуляризації':<23} | {mae_base:<8.4f} | {r2_base:<6.4f}")
print(f"{'З регуляризацією (L2)':<23} | {mae_l2:<8.4f} | {r2_l2:<6.4f}")

# завдання 4

plt.figure(figsize=(8, 8))

plt.scatter(y_test, pred_l2, color='blue', alpha=0.6, label='Передбачення моделі (L2)')

min_val = min(y_test.min(), pred_l2.min())
max_val = max(y_test.max(), pred_l2.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ідеальні передбачення (y = x)')

plt.title('Порівняння реальних та передбачених значень (Датасет Діабет)')
plt.xlabel('Реальні значення (y_test)')
plt.ylabel('Передбачені значення (y_pred)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig('diabetes_healthrisk_analysis.png', dpi=300, bbox_inches='tight')
plt.show()