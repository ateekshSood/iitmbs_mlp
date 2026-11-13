# %%

from pandas.core.common import random_state
import pandas as pd 


df = pd.read_csv("w4_5/GA_5_dataset.csv")

# %%

df.describe()
# %%

df.info()
# %%

from sklearn.model_selection import train_test_split

y = df['Credit_Limit']
x = df.drop(columns=['Credit_Limit'])



# %%


x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 42)


# %%

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(x_train , y_train)

# %%
from sklearn.metrics import r2_score

print(r2_score(y_test , lin_reg.predict(x_test)))
# %%

lin_reg.coef_
# %%

from sklearn.linear_model import Ridge

ridge_reg = Ridge(solver = "sag" , tol = 0.0005 , random_state=42)

ridge_reg.fit(x_train , y_train)

# %%

print(r2_score(y_test , ridge_reg.predict(x_test)))
# %%

print(ridge_reg.intercept_)
# %%


from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha = 100 , random_state= 42)

lasso_reg.fit(x_train , y_train)

# %%

print(r2_score(y_test , lasso_reg.predict(x_test)))
# %%

ridge_reg.coef_

# %%


print(len([coef for coef in ridge_reg.coef_ if coef >= -1 and coef <= 1]))
# %%


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors= 10 , p = 1)
knn.fit(x_train , y_train )


# %%

from sklearn.metrics import root_mean_squared_error

root_mean_squared_error(y_test , knn.predict(x_test))
# %%


from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth= 10 ,min_samples_split=6 , min_samples_leaf=6 , random_state=42)

tree.fit(x_train , y_train)

root_mean_squared_error(y_test , tree.predict(x_test))
# %%

from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV


ada = AdaBoostRegressor(random_state=42)

params = {
    "n_estimators" : [10,50,100,200,500],
    "learning_rate" : [0.1,0.5,1,2]
}

cross_val = GridSearchCV(ada , params , cv = 4)

cross_val.fit(x_train , y_train)

# %%


r2_score(y_test , cross_val.best_estimator_.predict(x_test))
# %%

cross_val.best_params_
