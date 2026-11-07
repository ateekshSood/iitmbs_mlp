# %%

from re import S
from jedi.third_party.typeshed.stubs.docutils.docutils.parsers.rst.states import Line
import pandas as pd

# %%

df = pd.read_csv("w4_5/california_housing_train.csv")
df.head(3)
# %%

df.isna().sum()
# %%
df.info()
# %%
df.describe()
# %%

from sklearn.model_selection import train_test_split

x = df.drop('median_house_value' , axis=1)
y = df['median_house_value']

x_train , x_val , y_train , y_val = train_test_split(x , y , random_state = 42 , test_size = 0.25)
# %%

print(x_train.shape)
print(x_val.shape)
# %%

import matplotlib.pyplot as plt
import seaborn as sns

df.hist(figsize=(12 , 10) , edgecolor="black" , bins= 30)
plt.show()
# %%

x_train.sample(4)
# %%

print(x_train.columns)

# %%

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

ct = ColumnTransformer([
    ('mms' , MinMaxScaler() , ['housing_median_age' , 'total_rooms' , 'population' , 'households']),
    ('ss' , StandardScaler() , ['longitude' , 'latitude']),
    ('mas' , MaxAbsScaler() , ['median_income'])
    ],
    verbose_feature_names_out=False,
    remainder = "passthrough"
).set_output(transform =  'pandas')

print(ct)
# %%

x_train_transform = ct.fit_transform(x_train)
x_train_transform.sample(4)
# %%

lin_reg = LinearRegression()
lin_reg.fit(x_train_transform , y_train)
# %%

from sklearn.metrics import r2_score


y_pred = lin_reg.predict(
    ct.transform(x_val)
)

print(r2_score( y_val , y_pred))
# %%

lin_reg_another = LinearRegression()
lin_reg_another.fit(x_train , y_train)

print(r2_score(y_val , lin_reg_another.predict(x_val) ) )
# %%


print(lin_reg.coef_)
# %%
print(lin_reg.intercept_)
# %%

another_lin_reg = LinearRegression(fit_intercept=False)
another_lin_reg.fit(x_train_transform , y_train)
# %%

another_lin_reg.intercept_
# %%

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor()

sgd_reg.fit(x_train_transform , y_train)

# %%
from sklearn.compose import TransformedTargetRegressor

x_val_transformed = ct.transform(x_val)

sgd_model = TransformedTargetRegressor(
    regressor=SGDRegressor(random_state = 42),
    transformer=StandardScaler()
)

sgd_model.fit(x_train_transform , y_train)


print(r2_score(y_val , sgd_model.predict(x_val_transformed)))
