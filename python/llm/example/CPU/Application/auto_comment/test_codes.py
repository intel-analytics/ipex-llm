import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.randint(0, 100, (3, 10))
df = pd.DataFrame(data)
row_means = df.mean(axis=1)
filtered_df = df.where(df >= row_means[:, np.newaxis], 0)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i, row in enumerate(df.values):
    axs[i].hist(row, bins=5, alpha=0.7, label=f'{i + 1}')
plt.tight_layout()
plt.show()