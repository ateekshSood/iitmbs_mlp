# %%

from IPython.core.pylabtools import figsize
from prompt_toolkit.input import PipeInput
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
# %%
plt.scatter(x['Longitude'] , x['Latitude'] ,s = 2 , alpha=0.3 , c='tab:blue')
plt.axis('off')
plt.show()


# %%
# x['MedHouseVal'] = y

# sns.pairplot(data = x , hue = 'MedHouseVal' , palette = "viridis")
#


y.hist(figsize=(12,10) , bins=30, edgecolor='black')
plt.title(y.name)
plt.show()
# %%

import numpy as np
from sklearn.model_selection import train_test_split

# %%

np.random.seed(306)

# %%

features , labels = fetch_california_housing(as_frame = True  , return_X_y = True)
# %%

x_train_full , x_test , y_train_full , y_test = train_test_split(features , labels , test_size = 0.2 , random_state = 42)
x_train , x_val , y_train , y_val = train_test_split(x_train_full , y_train_full , test_size = 0.2 , random_state = 42)
# %%

print(x_train.size)
print(x_test.size)
print(x_val.size)

# %%

# linear reg using normal eqn


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

lin_reg_pipeline = make_pipeline(
    [StandardScaler()],
    [LinearRegression()]
)

# %%

