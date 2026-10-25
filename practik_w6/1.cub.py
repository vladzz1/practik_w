import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) # кожен раз будуть однакові дані

fig, ax = plt.subplots(1,3, figsize=(15,4))

for i,n in enumerate([50,1000,100000]):
    rolls = np.random.randint(1,7,size=n)
    values, counts = np.unique(rolls, return_counts=True)
    ax[i].bar(values, counts/n, color='steelblue', edgecolor='black')
    ax[i].axhline(1/6, color='red', linestyle='--',label="Теоретична імовірність 1/6")
    ax[i].set_title(f"{n} підкидань кубика")
    ax[i].set_xlabel("Грані кубика")
    ax[i].set_ylabel("Частота")
    ax[i].legend()
    ax[i].set_ylim(0,0.3)
plt.suptitle("Випадкова величина: Результат підкидання кубика\n"
             "Чим більше спроб - тим ближче до теоретичного розподілу")
plt.tight_layout()
plt.show()