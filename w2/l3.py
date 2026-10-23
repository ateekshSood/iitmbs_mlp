# %%

from sklearn.datasets import load_iris

# %%

x , y = load_iris(as_frame=True , return_X_y = True)
# %%
print(x.shape)
print(y.shape)
# %%
print(type(x))
# %%
print(y.unique())
# %%
x.columns
# %%
data = load_iris()
# %%
print(type(data))
# %%
x = data.data
y = data.target
# %%
print(type(x))
print(type(y))
# %%

data.feature_names
# %%

from sklearn.datasets import load_diabetes

data = load_diabetes()
# %%
print(data.DESCR)
# %%
data.feature_names
# %%
print(data.data.shape)
# %%
import pandas as pd

# %%
print(pd.DataFrame(data.data).head(4))
# %%

print(data.data[:4])
# %%
print(data.target.shape)
# %%
print(data.target[:4])
# %%
print(data.feature_names)
# %%

from sklearn.datasets import load_digits

digits = load_digits()
# %%
print(digits.feature_names)
print(digits.data.shape)
print(digits.target.shape)
# %%

print(type(digits.data))
# %%

x , y  = digits.data , digits.target
# %%

import matplotlib.pyplot as plt

plt.imshow(x[1 , :].reshape(8 , 8) , cmap="binary")
plt.axis(False)
plt.show()

# %%
from sklearn.datasets import fetch_openml

# Fetch the full MNIST dataset (returns 70,000 samples)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Normalize pixel values from 0-255 to 0-1
X = X / 255.0

# Split into standard training and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


# %%