# %%

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# %%

x , y = make_blobs(n_samples = 10 , centers = 3 , n_features = 2 , random_state = 42)
plt.scatter(x[: , 0] , x[: , 1] , c=y , cmap="viridis")
plt.grid()
plt.show()
