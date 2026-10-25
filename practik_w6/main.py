import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, median_abs_deviation, zscore

# -- практична --
# завдання 1

file_path = 'transactions.csv'

try:
    df = pd.read_csv(file_path)
    print("Файл успішно завантажено!")
except FileNotFoundError:
    print(f"Помилка: Файл за адресою '{file_path}' не знайдено.")

    np.random.seed(42)
    n = 200
    data = {
        'transaction_id': range(1001, 1001 + n),
        'customer_id': np.random.randint(5000, 6000, size=n),
        'amount': np.random.exponential(scale=150, size=n) + 10,
        'status': np.random.choice(['completed', 'pending', 'failed'], size=n, p=[0.85, 0.10, 0.05])
    }
    df = pd.DataFrame(data)
    df.loc[df.sample(frac=0.05).index, 'amount'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'status'] = None

print("\n--- Перші 5 рядків таблиці ---")
print(df.head())

print("\n--- Інформація про стовпці та типи даних ---")
print(df.info())

print("\n--- Кількість пропусків у кожному стовпчику ---")
print(df.isnull().sum())

def calculate_amount_summary(data_series, title_suffix=""):
    print(f"\n--- Статистичне зведення для 'amount' {title_suffix} ---")

    count_val = data_series.count()
    mean_val = data_series.mean()
    median_val = data_series.median()
    min_val = data_series.min()
    max_val = data_series.max()
    
    q25 = np.nanpercentile(data_series, 25)
    q75 = np.nanpercentile(data_series, 75)
    p90 = np.nanpercentile(data_series, 90)
    p95 = np.nanpercentile(data_series, 95)
    
    summary_df = pd.Series({
        'Кількість (Count)': count_val,
        'Середнє (Mean)': mean_val,
        'Медіана (Median)': median_val,
        'Мінімум (Min)': min_val,
        'Максимум (Max)': max_val,
        '25-й квартиль (Q1)': q25,
        '75-й квартиль (Q3)': q75,
        '90-й перцентиль (P90)': p90,
        '95-й перцентиль (P95)': p95
    })
    print(summary_df.to_string())
    return summary_df

calculate_amount_summary(df['amount'], "(ДО видалення пропусків)")

def plot_distributions(data_series, title_modifier=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(data_series.dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f"Гістограма розподілу сум замовлень {title_modifier}")
    axes[0].set_xlabel("Сума замовлення (amount)")
    axes[0].set_ylabel("Частота / Кількість")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1].boxplot(data_series.dropna(), orientation='horizontal', patch_artist=True, boxprops=dict(facecolor='lightgreen', color='black'), medianprops=dict(color='red', linewidth=2))
    axes[1].set_title(f"Коробчата діаграма 'amount' {title_modifier}")
    axes[1].set_xlabel("Сума замовлення (amount)")
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

plot_distributions(df['amount'], "(із пропусками)")

df_cleaned = df.dropna(subset=['amount']).copy()

print("\n==============================================")
print("ПІСЛЯ ВИДАЛЕННЯ ПРОПУСКІВ У СТОВПЦІ 'AMOUNT'")
print("==============================================")
print(f"Нова розмірність таблиці: {df_cleaned.shape}")
print(f"Кількість пропусків в amount тепер: {df_cleaned['amount'].isnull().sum()}")

calculate_amount_summary(df_cleaned['amount'], "(ПІСЛЯ видалення пропусків)")

plot_distributions(df_cleaned['amount'], "(після очищення)")

# завдання 2

np.random.seed(42)
base_data = np.random.normal(loc=50, scale=10, size=80)
base_data = np.clip(base_data, 1, None)

outliers = np.array([250, 310, 285, 400, 350])
full_array = np.concatenate([base_data, outliers])

df = pd.DataFrame(full_array, columns=['values'])

mean_val = df['values'].mean()
median_val = df['values'].median()
std_val = df['values'].std()

q25 = df['values'].quantile(0.25)
q75 = df['values'].quantile(0.75)
iqr_val = q75 - q25

mad_val = median_abs_deviation(df['values'])

skew_val = skew(df['values'])
kurt_val = kurtosis(df['values'])

print("--- Статистичні показники розподілу ---")
print(f"Середнє (Mean):                 {mean_val:.2f}")
print(f"Медіана (Median):               {median_val:.2f}")
print(f"Стандартне відхилення (Std):    {std_val:.2f}")
print(f"Міжквартильний розмах (IQR):    {iqr_val:.2f}")
print(f"Медіанне відхилення (MAD):      {mad_val:.2f}")
print(f"Коефіцієнт асиметрії (Skew):    {skew_val:.2f} (значна правостороння асиметрія)")
print(f"Коефіцієнт ексцесу (Kurtosis):  {kurt_val:.2f} (гостровершинний розподіл)")

lower_bound = q25 - 1.5 * iqr_val
upper_bound = q75 + 1.5 * iqr_val

outliers_condition = (df['values'] < lower_bound) | (df['values'] > upper_bound)
outliers_count = df[outliers_condition].shape[0]
total_count = df.shape[0]
outliers_share = outliers_count / total_count

print("\n--- Аналіз викидів за правилом Boxplot (Tukey's fences) ---")
print(f"Нижня межа 'вус':               {lower_bound:.2f}")
print(f"Верхня межа 'вус':               {upper_bound:.2f}")
print(f"Кількість викидів за 'вусами':  {outliers_count} з {total_count}")
print(f"Частка викидів у вибірці:        {outliers_share * 100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['values'], bins=25, color='salmon', edgecolor='black', alpha=0.7)
axes[0].axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f"Середнє ({mean_val:.1f})")
axes[0].axvline(median_val, color='green', linestyle='-', linewidth=2, label=f"Медіана ({median_val:.1f})")
axes[0].set_title("Гістограма розподілу (з аномаліями)")
axes[0].set_xlabel("Значення")
axes[0].set_ylabel("Частота")
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

axes[1].boxplot(df['values'], orientation='horizontal', patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'), medianprops=dict(color='green', linewidth=2), flierprops=dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none'))
axes[1].set_title("Коробчата діаграма та точки-викиди")
axes[1].set_xlabel("Значення")
axes[1].grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# завдання 3

np.random.seed(24)

ratings = np.random.choice(
    a=[1, 2, 3, 4, 5], 
    size=200, 
    p=[0.05, 0.10, 0.20, 0.40, 0.25]
)

df = pd.DataFrame({'rating': ratings})

freq_table = df['rating'].value_counts().sort_index().to_frame(name='Абсолютна частота')

freq_table['Відносна частота'] = df['rating'].value_counts(normalize=True).sort_index()

freq_table['Накопичена частка'] = freq_table['Відносна частота'].cumsum()

print("--- Частотна таблиця для оцінок 'rating' ---")
print(freq_table)

mean_rating = df['rating'].mean()
median_rating = df['rating'].median()
mode_rating = df['rating'].mode()[0]

print("\n--- Центральні тенденції розподілу ---")
print(f"Мода (Найчастіша оцінка):  {mode_rating}")
print(f"Медіана (Центральна точка): {median_rating}")
print(f"Середня оцінка (Mean):      {mean_rating:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(freq_table.index, freq_table['Абсолютна частота'], color='teal', edgecolor='black', alpha=0.7)
axes[0].set_title("Стовпчаста діаграма частот оцінок")
axes[0].set_xlabel("Оцінка (Rating)")
axes[0].set_ylabel("Кількість респондентів")
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

for index, value in zip(freq_table.index, freq_table['Абсолютна частота']):
    axes[0].text(index, value + 1, str(value), ha='center', va='bottom', fontsize=10)

x_cum = [0] + list(freq_table.index)
y_cum = [0] + list(freq_table['Накопичена частка'])

axes[1].plot(x_cum, y_cum, marker='o', color='crimson', linewidth=2, label='Накопичена частка')
axes[1].fill_between(x_cum, y_cum, color='crimson', alpha=0.1)
axes[1].set_title("Графік накопичених часток (Кумулята)")
axes[1].set_xlabel("Оцінка (Rating)")
axes[1].set_ylabel("Частка (від 0.0 до 1.0)")
axes[1].set_xlim(0.5, 5.5)
axes[1].set_ylim(0, 1.05)
axes[1].set_xticks([1, 2, 3, 4, 5])
axes[1].grid(linestyle='--', alpha=0.5)

for x, y in zip(freq_table.index, freq_table['Накопичена частка']):
    axes[1].text(x, y + 0.02, f"{y:.1%}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# завдання 4

np.random.seed(42)
n_people = 320

heights = np.random.normal(loc=175, scale=7, size=n_people)

extreme_outliers = np.array([135.0, 142.0, 202.0, 208.0, 215.0])
full_heights = np.concatenate([heights, extreme_outliers])

df = pd.DataFrame({'height': full_heights})

percentiles_list = [5, 25, 50, 75, 95]
calculated_percentiles = np.percentile(df['height'], percentiles_list)

print("--- Перцентилі для стовпця 'height' ---")
for p, val in zip(percentiles_list, calculated_percentiles):
    print(f"{p}-й перцентиль: {val:.2f} см")

df['height_z'] = zscore(df['height'])

print("\n--- Перші 5 рядків таблиці з Z-оцінками ---")
print(df.head())

outliers_condition = df['height_z'].abs() > 3
df_outliers = df[outliers_condition]

print("\n--- Аналіз викидів за методом Z-оцінок (|z| > 3) ---")
print(f"Загальна кількість спостережень: {df.shape[0]}")
print(f"Кількість виявлених викидів:     {df_outliers.shape[0]}")

if df_outliers.shape[0] > 0:
    print("\nЗнайдені аномальні спостереження:")
    print(df_outliers)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['height'], bins=25, color='darkseagreen', edgecolor='black', alpha=0.7)
axes[0].set_title("Гістограма розподілу зросту (height)")
axes[0].set_xlabel("Зріст (см)")
axes[0].set_ylabel("Кількість людей")
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

axes[0].axvline(df['height'].mean(), color='red', linestyle='--', linewidth=2, label=f"Середнє ({df['height'].mean():.1f} см)")
axes[0].legend()

axes[1].hist(df['height_z'], bins=25, color='cadetblue', edgecolor='black', alpha=0.7)
axes[1].set_title("Гістограма розподілу Z-оцінок (height_z)")
axes[1].set_xlabel("Z-оцінка (кількість стандартних відхилень від середнього)")
axes[1].set_ylabel("Кількість людей")
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

axes[1].axvline(-3, color='crimson', linestyle=':', linewidth=2, label="Критична межа (Z = -3)")
axes[1].axvline(3, color='crimson', linestyle=':', linewidth=2, label="Критична межа (Z = 3)")
axes[1].legend()

plt.tight_layout()
plt.show()

# -- домашня --
# завдання 1

np.random.seed(100)
n_days = 30

date_range = pd.date_range(start='2026-09-01', periods=n_days, freq='D')

users = np.random.randint(1000, 2500, size=n_days)
sessions = users * np.random.uniform(1.2, 1.5, size=n_days) + np.random.randint(-50, 50, size=n_days)
revenue = sessions * np.random.uniform(15, 25, size=n_days) + np.random.normal(0, 2000, size=n_days)
revenue = np.clip(revenue, 0, None)

df = pd.DataFrame({
    'date': date_range,
    'users': users,
    'sessions': sessions,
    'revenue': revenue
})

correlation_matrix = df[['users', 'sessions', 'revenue']].corr()

print("--- Кореляційна матриця метрик ---")
print(correlation_matrix)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(df['users'], df['sessions'], color='purple', alpha=0.7, edgecolors='w', s=50)
axes[0].set_title(f"Зв'язок: Users та Sessions\nCorr: {correlation_matrix.loc['users', 'sessions']:.2f}")
axes[0].set_xlabel("Користувачі (users)")
axes[0].set_ylabel("Сесії (sessions)")
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].scatter(df['users'], df['revenue'], color='darkblue', alpha=0.7, edgecolors='w', s=50)
axes[1].set_title(f"Зв'язок: Users та Revenue\nCorr: {correlation_matrix.loc['users', 'revenue']:.2f}")
axes[1].set_xlabel("Користувачі (users)")
axes[1].set_ylabel("Виручка (revenue, грн)")
axes[1].grid(True, linestyle='--', alpha=0.5)

axes[2].scatter(df['sessions'], df['revenue'], color='darkgreen', alpha=0.7, edgecolors='w', s=50)
axes[2].set_title(f"Зв'язок: Sessions та Revenue\nCorr: {correlation_matrix.loc['sessions', 'revenue']:.2f}")
axes[2].set_xlabel("Сесії (sessions)")
axes[2].set_ylabel("Виручка (revenue, грн)")
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(df['date'], df['revenue'], marker='o', color='crimson', linewidth=2, label='Щоденна виручка')
plt.title("Динаміка виручки (revenue) за 30 днів")
plt.xlabel("Дата")
plt.ylabel("Виручка (грн)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# завдання 2

np.random.seed(42)
n_samples = 150

converted_A = np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12])
converted_B = np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18])

df_A = pd.DataFrame({'group': 'A', 'converted': converted_A})
df_B = pd.DataFrame({'group': 'B', 'converted': converted_B})

df = pd.concat([df_A, df_B], ignore_index=True)

stats = df.groupby('group')['converted'].agg(['count', 'sum'])
stats['conversion_rate'] = stats['sum'] / stats['count']

conv_A = stats.loc['A', 'conversion_rate']
conv_B = stats.loc['B', 'conversion_rate']

abs_diff = conv_B - conv_A
rel_diff = (conv_B - conv_A) / conv_A

print("--- Статистика по групах ---")
print(f"Конверсія групи А: {conv_A:.2%}")
print(f"Конверсія групи B: {conv_B:.2%}")
print(f"Абсолютна різниця (B - A): {abs_diff:.2%} (на стільки відсоткових пунктів)")
print(f"Відносна зміна (відсоткове покращення): {rel_diff:.2%}")

z_score = 1.96

stats['ci_error'] = z_score * np.sqrt((stats['conversion_rate'] * (1 - stats['conversion_rate'])) / stats['count'])

print("\n--- 95% Довірчі інтервали для конверсії ---")
print(f"Група А: від {(conv_A - stats.loc['A', 'ci_error']):.2%} до {(conv_A + stats.loc['A', 'ci_error']):.2%}")
print(f"Група B: від {(conv_B - stats.loc['B', 'ci_error']):.2%} до {(conv_B + stats.loc['B', 'ci_error']):.2%}")

plt.figure(figsize=(8, 6))

bars = plt.bar(
    stats.index, 
    stats['conversion_rate'], 
    yerr=stats['ci_error'], 
    capsize=10,
    color=['#3498db', '#2ecc71'], 
    edgecolor='black', 
    alpha=0.8,
    error_kw=dict(ecolor='dimgray', lw=2, capthick=2)
)

plt.title("Порівняння конверсії в групах A та B (з 95% CI)", fontsize=14)
plt.xlabel("Група експерименту", fontsize=12)
plt.ylabel("Конверсія (Conversion Rate)", fontsize=12)
plt.ylim(0, max(conv_A, conv_B) + max(stats['ci_error']) + 0.05)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2., 
        height / 2,
        f"{height:.2%}", 
        ha='center', 
        va='center', 
        color='white', 
        fontweight='bold', 
        fontsize=12
    )

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# завдання 3

np.random.seed(42)
population_size = 55000

population = np.random.exponential(scale=10, size=population_size)

pop_mean = np.mean(population)
pop_std = np.std(population)

print("--- Генеральна сукупність (Експоненційний розподіл) ---")
print(f"Істинне середнє сукупності:           {pop_mean:.4f}")
print(f"Істинне стандартне відхилення (std): {pop_std:.4f}\n")

n_samples_to_draw = 1500 

n_values = [5, 50]
sample_means_dict = {}

for n in n_values:
    means = []
    for _ in range(n_samples_to_draw):
        sample = np.random.choice(population, size=n, replace=False)
        means.append(np.mean(sample))
    
    sample_means_dict[n] = means
    
    print(f"--- Статистика вибіркових середніх для n = {n} ---")
    print(f"Середнє вибіркових середніх:          {np.mean(means):.4f} (близьке до {pop_mean:.1f})")
    print(f"Стандартне відхилення (станд. помилка): {np.std(means):.4f} (має дорівнювати ~ {pop_std / np.sqrt(n):.4f})")
    print("-" * 50)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(population, bins=50, color='coral', edgecolor='black', alpha=0.7, density=True)
axes[0].set_title("Початкова сукупність\n(Експоненційний розподіл)")
axes[0].set_xlabel("Значення")
axes[0].set_ylabel("Щільність")
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].hist(sample_means_dict[5], bins=30, color='gold', edgecolor='black', alpha=0.7, density=True)
axes[1].set_title(f"Розподіл середніх при n = 5\n(std = {np.std(sample_means_dict[5]):.2f})")
axes[1].set_xlabel("Вибіркове середнє")
axes[1].grid(True, linestyle='--', alpha=0.5)

axes[2].hist(sample_means_dict[50], bins=30, color='yellowgreen', edgecolor='black', alpha=0.7, density=True)
axes[2].set_title(f"Розподіл середніх при n = 50\n(std = {np.std(sample_means_dict[50]):.2f})")
axes[2].set_xlabel("Вибіркове середнє")
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# завдання 4

np.random.seed(42)
n_days = 90

date_range = pd.date_range(start='2026-07-01', periods=n_days, freq='D')

trend = np.linspace(100, 250, n_days)
noise = np.random.normal(scale=30, size=n_days)
sales = trend + noise
sales = np.clip(sales, 10, None)

df = pd.DataFrame({
    'date': date_range,
    'sales': sales
})

window_size = 7

df['sales_rolling_mean'] = df['sales'].rolling(window=window_size).mean()
df['sales_rolling_std'] = df['sales'].rolling(window=window_size).std()

print("--- Перші 12 рядків таблиці з ковзними метриками ---")
print(df.head(12).to_string(index=False))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(df['date'], df['sales'], color='lightgray', marker='o', alpha=0.7, label='Вихідні продажі (щоденні)')
ax1.plot(df['date'], df['sales_rolling_mean'], color='crimson', linewidth=2.5, label=f'Ковзне середнє ({window_size}-денне)')
ax1.set_title("Аналіз щоденних продажів та згладжений тренд", fontsize=14)
ax1.set_ylabel("Продажі (sales)")
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.plot(df['date'], df['sales_rolling_std'], color='darkblue', linewidth=2, linestyle='-', label=f'Ковзне std ({window_size}-денне)')
ax2.set_title("Динаміка ковзного стандартного відхилення (волатильність продажів)", fontsize=14)
ax2.set_xlabel("Дата")
ax2.set_ylabel("Стандартне відхилення (std)")
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()