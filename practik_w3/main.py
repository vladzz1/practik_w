import numpy as np #Бібліотека для роботи з масивами C++

myarray = np.array([23, 56, 1, 4, 90, 9])
print(myarray) #виводимо масив
print(" -- Working matrix -- ")
myArray2 = np.array([[12, 3, 4], [8, 9, 5]])
print(myArray2)
print("-- Нулевий масив --")
zeros = np.zeros((3, 3))
print(zeros)
print(" -- Масив із одиничок -- ")
ones = np.ones((3, 3))
print(ones)
print(" -- Масив із діапазону значень -- ")
arr = np.arange(0, 10, 2)
print(arr)
print("-- Поділ проміжку на частини --")
arr = np.linspace(0, 2, 10)
print(arr)

print("-- Масив може зберігати будь-які дані --")
myArray = np.array([[2, 3, "Сало"], [4, False, 6]])
print(myarray)
print("-- Про масив можна отримати наступні дані --")
print("Shape", myarray.shape) #Розмір масиву
print("ndim", myarray.ndim) #Кількість вимірів у масиві
print("dtype", myarray.dtype) #тип даних у елемента
print("size", myarray.size) #кількість усіх елементів

print("-- Зріз елементів у масиві --")
myList = np.array([34, 56, 77, 3, -6, 32, 345, 12, -8])
print("First element", myList[0])
print("Last element", myList[-1]) #Останній елемент списку
print("1:4", myList[1:4]) # не включно з 4 - вказуємо індекс
print(":3", myList[:3]) # 3 з початку
print("2:", myList[2:]) # від 2 індексу і до кінця
print("::2", myList[::2]) #кожен другий елемент списку
print("-- Зрізи можна робити з 2 мірними масивами --")
my2d = np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
print("My 2d", my2d)
print("[1:, :2]", my2d[1:, :2])

print("-- Операції над масивами --")
a = np.array([2, 3, 4, 6, 1])
b = np.array([-4, 4, 5, 9, -6])
print("a:", a)
print("b:", b)
print("a + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", a / b)
print("a // b:", a // b) # ділення без остачі із заокругленням до меншого
print("a % b:", a % b)
print("a == b:", a == b)
print("a != b:", a != b)

print(" -- Статичні методи в np -- ")
data = np.array([2, 3, 5, 6, 7, 2, 3, 90, 54 ,3 ,4])
print("data", data)
print("mean", np.mean(data)) # Середнє арифметичне
print("median", np.median(data)) # Медіана більше точне середнє
print("min", np.min(data))
print("max", np.max(data))

# -- практична --
# завдання 1

print("-- практична --")
array = np.array([23, 45, 12, 56, 67, 24, 61, 36, 15, 50])
print(array)
print("мінімальне:", np.min(array))
print("максимальне:", np.max(array))
print("середнє:", np.median(array))
print(array)
sort_array = np.sort(array)
print(sort_array)

# завдання 2

matrix = np.array([[3, 5, 7, 14], [3, 5, 15, 7], [34, 32, 12, 14], [23, 25, 12, 6]])
for row in matrix:
    print(" ".join(f"{num:>4}" for num in row))
print("сума стовпців:", matrix.sum(axis=0))
print("сума рядків:", matrix.sum(axis=1))
print("середнє значення стовпців:", np.median(matrix.sum(axis=0)))
print("середнє значення рядків:", np.median(matrix.sum(axis=1)))

# завдання 3

matrix = np.array([[2, 0, 11, 5, 6], [3, 6, 14, 3, 1], [3, 6, 7, 5, 9]])
print(matrix)
transposed = matrix.T
print(transposed)
reshaped_matrix = matrix.reshape(5, 3)
print(reshaped_matrix)

# завдання 4

matrix = np.round(np.random.rand(10, 10), 2)
print(matrix)
min_val = np.min(matrix)
max_val = np.max(matrix)
min_flat_idx = np.argmin(matrix)
max_flat_idx = np.argmax(matrix)
min_idx = np.unravel_index(min_flat_idx, matrix.shape)
max_idx = np.unravel_index(max_flat_idx, matrix.shape)
row_min, col_min = map(int, min_idx)
row_max, col_max = map(int, max_idx)
print(f"Мінімум: {min_val} на позиції ({row_min}, {col_min}) (рядок {row_min}, стовпець {col_min})")
print(f"Максимум: {max_val} на позиції ({row_max}, {col_max}) (рядок {row_max}, стовпець {col_max})")

# завдання 5

array = np.random.randint(-10, 11, size=15)
print(array)
positive_numbers_mask = array >= 0
even_numbers_mask = array % 2 == 0
positive_numbers = array[positive_numbers_mask]
even_numbers = array[even_numbers_mask]
print("додатні:", positive_numbers)
print("парні:", even_numbers)

# завдання 6

array1 = np.random.randint(1, 6, size=5)
array2 = np.random.randint(6, 11, size=5)
matrix_2x5 = np.concatenate((array1[np.newaxis, :], array2[np.newaxis, :]), axis=0)
print("матриця 2x5:")
print(matrix_2x5)
matrix_5x2 = matrix_2x5.reshape(5, 2)
print("матриця 5x2:")
print(matrix_5x2)

# -- домашня --
# завдання 1

print("-- домашня --")
array = np.random.randint(0, 51, size=20)
print(array)
while(True):
    try:
        threshold_value = int(input("введіть порогове значення: "))
        break
    except Exception as e:
        print(f"\033[31m{e}\033[0m")
count_number = np.sum(array > threshold_value)
print(f"кількість чисел більших за {threshold_value}:", count_number)
max_index = np.argmax(array)
max_value = array[max_index]
print(f"максимальний елемент: {max_value}, позиція: {max_index}")
sort_array = np.sort(array)[::-1]
print("відсортований масив:", sort_array)

# завдання 2

while(True):
    try:
        start_value = int(input("введіть нижню межу діапазону: "))
        end_value = int(input("введіть верхню межу діапазону: "))
        break
    except Exception as e:
        print(f"\033[31m{e}\033[0m")
if start_value > end_value:
    temp = start_value
    start_value = end_value
    end_value = temp
matrix = np.random.randint(start_value, end_value, size=(5, 5))
print(matrix)
print("головна діагональ:", np.diag(matrix))
print("сума головної діагоналі:", np.sum(np.diag(matrix)))
print("матриця з нулями:\n", np.tril(matrix))

# завдання 3

while(True):
    try:
        start_value = int(input("введіть нижню межу діапазону: "))
        end_value = int(input("введіть верхню межу діапазону: "))
        break
    except Exception as e:
        print(f"\033[31m{e}\033[0m")
if start_value > end_value:
    temp = start_value
    start_value = end_value
    end_value = temp
array = np.arange(start_value, end_value + 1)
full_sequence = np.zeros(30, dtype=int)
elements_to_copy = min(len(array), 30)
full_sequence[:elements_to_copy] = array[:elements_to_copy]
matrix = full_sequence.reshape(6, 5)
print(matrix)
print("суми рядків:", np.sum(matrix, axis=1))
print("максимуми стовпчиків:", np.max(matrix, axis=0))

# завдання 4

while(True):
    try:
        start_value = int(input("введіть нижню межу діапазону (відємне число): "))
        end_value = int(input("введіть верхню межу діапазону: "))
        break
    except Exception as e:
        print(f"\033[31m{e}\033[0m")
if start_value > end_value:
    temp = start_value
    start_value = end_value
    end_value = temp
if start_value >= 0:
    start_value = start_value * -1
array = np.random.randint(start_value, end_value, size=15)
print(array)
print("від'ємні числа:", array[array < 0])
array[array < 0] = 0
print(array)
print("сума нулів:", np.sum(array == 0))

# завдання 5

while(True):
    try:
        arr_len1 = int(input("введіть довжину першого масиву: "))
        arr_len2 = int(input("введіть довжину другого масиву: "))
        break
    except Exception as e:
        print(f"\033[31m{e}\033[0m")
if arr_len1 < 0:
    arr_len1 = arr_len1 * -1
if arr_len2 < 0:
    arr_len2 = arr_len2 * -1
array1 = np.random.randint(0, 11, size=arr_len1)
array1 = np.random.randint(10, 21, size=arr_len2)
print("масив 1:", array1)
print("масив 2:", array2)
print("об'єднаний масив:", np.concatenate((array1, array2)))
max_len = max(len(array1), len(array2))
arr1_padded = np.zeros(max_len, dtype=int)
arr2_padded = np.zeros(max_len, dtype=int)
arr1_padded[:len(array1)] = array1
arr2_padded[:len(array2)] = array2
print("поелементне додавання:", arr1_padded + arr2_padded)
print("поелементна різниця:", arr1_padded - arr2_padded)

# завдання 6

while(True):
    try:
        matrix_row = int(input("введіть кількість рядків матриці: "))
        matrix_col = int(input("введіть кількість стовпців матриці: "))
        break
    except Exception as e:
        print(f"\033[31m{e}\033[0m")
if matrix_row < 0:
    matrix_row = matrix_row * -1
if matrix_col < 0:
    matrix_col = matrix_col * -1
sequence = np.arange(1, matrix_row * matrix_col + 1)
matrix = sequence.reshape(matrix_row, matrix_col)
print("матриця з послідовно зростаючих чисел:\n", matrix)
while(True):
    try:
        matrix_row = int(input("введіть нову кількість рядків матриці: "))
        matrix_col = int(input("введіть нову кількість стовпців матриці: "))
        break
    except Exception as e:
        print(f"\033[31m{e}\033[0m")
if matrix_row < 0:
    matrix_row = matrix_row * -1
if matrix_col < 0:
    matrix_col = matrix_col * -1
if matrix_row * matrix_col == matrix.size:
    matrix = sequence.reshape(matrix_row, matrix_col)
    print("нова матриця з послідовно зростаючих чисел:\n", matrix)
    print("мінімуми рядків:", np.min(matrix, axis=1))
    print("максимуми рядків:", np.max(matrix, axis=1))
    print("сума всіх елементів матриці:", np.sum(matrix))
else:
    print("\033[31mнеможливо зробити матрицю таких розмірів\033[0m")