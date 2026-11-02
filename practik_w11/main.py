import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Створення штучного набору даних
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# print("Data shape: ", X.shape)
# print("Label shape: ", y.shape)

scaler = StandardScaler()

X = scaler.fit_transform(X)
# Потрібно дані перетворити в тензор для нейронної мережі
X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float).view(-1, 1)

# print(X_tensor.shape, y_tensor.shape)
#поділ даних на тестові і для тренування
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

#Побудова моделі для нейронної мережі
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10,32), #z=Wx+b
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid() #Використовуємо бінарну класифікацію
        )

    # Визначаємо фукнцію для того в якому порядку і як виконувати шари нейронної мережі
    def forward(self, x):
        return self.model(x)

model = MLPClassifier()

#оптимізаціто і фукнцію втрат
criterion = nn.BCELoss() # Бінарна корос етропія
optimizer = optim.SGD(model.parameters()) # Оптиізатор градієнтного спуска

epochs = 200 # За допомогою кількості епох можна порівнюємо результати навчання нашої моделі

for epoch in range(epochs):

    # переводимо модель у режим тренування
    model.train()

    #Обнуляємо попередній градієнт
    optimizer.zero_grad()

    # forward propagation
    outputs = model(X_train)

    #підраховуємо помилку (forward pass)
    loss = criterion(outputs, y_train)

    #підрахунок градієнтів
    loss.backward()

    #оновляємо параметри моделі
    optimizer.step()

    # на кожному етапі показуємо як проводиться навчання нейронної мережі
    with torch.no_grad():
        # переводимо модель у режим інференсу
        model.eval()
        # отримуємо прогнози
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        accuracy = ((val_outputs > 0.5) == y_test).float().mean()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

# -- практична --
# завдання 1

torch.manual_seed(42)

age_1d = torch.randint(18, 70, (5,), dtype=torch.float32)
print("1D Тензор (Вік клієнтів):\n", age_1d)

weight_height_2d = torch.cat([
    torch.randint(50, 100, (5, 1), dtype=torch.float32),
    torch.randint(150, 190, (5, 1), dtype=torch.float32)
], dim=1)
print("\n2D Тензор (Вага та Зріст для 5 клієнтів):\n", weight_height_2d)

historical_data_3d = torch.randint(50, 190, (2, 5, 2), dtype=torch.float32)
print("\n3D Тензор (Дані за 2 періоди):\n", historical_data_3d)

print("\n--- 2. Операції додавання, множення та requires_grad ---")

age_risk_weight = torch.tensor([0.5, 0.4, 0.6, 0.2, 0.8], requires_grad=True)
lifestyle_risk_factor = torch.tensor([1.2, 1.0, 1.5, 1.1, 1.3])

combined_risk = age_1d + lifestyle_risk_factor
print("Результат додавання (Вік + Фактор способу життя):\n", combined_risk)

conditional_risk_score = combined_risk * age_risk_weight
print("\nУмовний показник ризику (Поелементне множення):\n", conditional_risk_score)

print("\n--- 3. Перевірка обчислювального графу (grad_fn) ---")

print("Чи відстежуються градієнти для кінцевого тензора?:", conditional_risk_score.requires_grad)
print("Наявність grad_fn (функція градієнта):", conditional_risk_score.grad_fn)

# завдання 2

torch.manual_seed(42)

X = torch.randn(100, 1)

true_w = 3.5
true_b = 1.2
noise = torch.randn(100, 1) * 0.2
y = X * true_w + true_b + noise

model = nn.Linear(1, 1)
criterion = nn.MSELoss()

lr = 0.1 

initial_w = model.weight.item()
initial_b = model.bias.item()

initial_pred = model(X)
initial_loss = criterion(initial_pred, y).item()

print("--- Початковий стан моделі ---")
print(f"Вага (w): {initial_w:.4f}")
print(f"Зсув (b): {initial_b:.4f}")
print(f"Початкова помилка (MSE Loss): {initial_loss:.4f}\n")


print("--- Процес навчання (20 епох) ---")
for epoch in range(20):
    predictions = model(X)
    
    loss = criterion(predictions, y)
    
    loss.backward()
    
    with torch.no_grad():
        model.weight -= lr * model.weight.grad
        model.bias -= lr * model.bias.grad
        
        model.weight.grad.zero_()
        model.bias.grad.zero_()
        
    if (epoch + 1) % 5 == 0:
        print(f"Епоха {epoch + 1}/20 | Поточний Loss: {loss.item():.4f}")


print("\n--- Фінальний стан моделі після навчання ---")
final_w = model.weight.item()
final_b = model.bias.item()

final_pred = model(X)
final_loss = criterion(final_pred, y).item()

print(f"Фінальна вага (w): {final_w:.4f} (Очікувалось близько {true_w})")
print(f"Фінальний зсув (b): {final_b:.4f} (Очікувалось близько {true_b})")
print(f"Фінальна помилка (MSE Loss): {final_loss:.4f}")