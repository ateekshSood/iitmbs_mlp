from cmath import pi
import pandas as pd
# %%
df = pd.read_csv("w6/NPPE1_ModelBuilding3.csv")
# %%

df.columns
# %%
y = df['14']
x = df.drop(columns = ['14'])


# %%



from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.30 , random_state = 42)

# %%


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


ridge = Ridge(alpha = 10 , solver = 'saga' , tol = 1e-4 , random_state = 42)
ridge.fit(x_train , y_train)


print(r2_score(y_test , ridge.predict(x_test)))
# %%
import numpy as np
# %%

print(ridge.coef_)

np.argmax(ridge.coef_) 
# %%
np.argmin(ridge.coef_) 


# %%


from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

sgd = SGDRegressor(random_state=42)

sgd.fit(x_train , y_train)

params = {
    "penalty" : ['l1' , 'l2'],
    "alpha" : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "tol" :  [1e-4, 1e-3, 1e-2, 1e-1]
}

grid = GridSearchCV(sgd , params , cv = 5 , scoring = 'neg_mean_absolute_error')
grid.fit(x_train , y_train)
# %%

final_model = grid.best_estimator_

# %%
grid.best_params_

# %%
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test , final_model.predict(x_test)))
# %%

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

pipeline = make_pipeline(
    PCA(),
    Lasso()
)

# pipeline.fit(x_train , y_train)

params = {
    "pca__n_components" : [0.9 , 0.95],
    "lasso__alpha" : [10 , 1 , 0.01 , 0.001]
}

grid = GridSearchCV(pipeline , params , scoring = 'neg_mean_absolute_error' , cv=5 , n_jobs=-1)

grid.fit(x_train , y_train)
# %%

print(r2_score(y_test , grid.best_estimator_.predict(x_test)))
# %%

print(grid.best_estimator_[# %%
    0].explained_variance_[0])


# %%

from sklearn.preprocessing import PolynomialFeatures

pipeline = make_pipeline(
    PolynomialFeatures(interaction_only=False , degree=2),
    Lasso(alpha=1 , warm_start=True , random_state=0)
)

pipeline.fit(x_train , y_train)

print(r2_score(y_test , pipeline.predict(x_test)))
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

rfe = RFE(estimator=LinearRegression() , n_features_to_select=1)
rfe.fit(x_train , y_train)
# %%
