from sklearn.datasets import load_digits

# %%

digits = load_digits(as_frame=True)

# %%

df  , y = digits.data , digits.target 


# %%

df.head(3)

# %%

df.shape

# %%

import matplotlib.pyplot as plt

plt.imshow(df.iloc[0].to_numpy().reshape(8 , 8 , -1) , cmap="binary")
plt.axis(False)
plt.show()
# %%


# main is mnist init 

from sklearn.datasets import fetch_openml


x , y = fetch_openml('mnist_784' , version = 1 , return_X_y = True)
# %%

x.head(3)
# %%

print(784 ** (0.5))
# %%

x = x.to_numpy()
y = y.to_numpy()

# %%
plt.imshow(x[0].reshape(28 , 28 , -1) , cmap="binary")
plt.axis('off')
plt.show()
# %%

import numpy as np

print(np.unique(y))
# %%


fig , ax = plt.subplots(3 ,3 , figsize=(12 , 10))

ax = ax.flatten()

for i in range(9):

    ax[i].imshow(x[i].reshape(28 , 28 , -1) , cmap='binary')
    ax[i].axis('off')
    ax[i].set_title(y[i])

plt.subplots_adjust()
plt.show()
# %%


from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = x[:60000] , x[60000:] , y[:60000] , y[60000:]

# %%
