# %%

from IPython.core.pylabtools import figsize
from sklearn.datasets import fetch_california_housing

# %%


x , y = fetch_california_housing(as_frame = True , return_X_y = True)
# %%


print(x.columns)
# %%

print(y.head(4))
# %%


print(x.info())
# %%

print(x.isna().sum())
# %%

print(x.head(4))
# %%
print(x.describe())
# %%

import matplotlib.pyplot as plt
import seaborn as sns
# %%


x.hist(figsize=(12,10) , bins = 30 , edgecolor='black')
plt.grid()
plt.subplots_adjust(hspace=0.7 , wspace=0.4)
plt.show()

# %%

print(len(x[x['Population'] < 10]))


