# %%
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np

# %%

df = pd.read_csv("w4_5/Clean_Dataset.csv")
# %%


df.describe()
# %%
df.info()
# %%
df.columns
# %%
df.drop('Unnamed: 0' , axis=1 , inplace=True)
# %%
y = df['price']
x = df.drop(columns = ['price'])

# %%
df.columns
# %%

print(df['airline'].dtype)
# %%

from sklearn.model_selection import train_test_split

x_train  , x_val , y_train , y_val = train_test_split(x , y , test_size = 0.20 , random_state = 42)



# %%

x_train.columns
# %%


columns_category = {
    "Numerical" : [],
    "Categorical" : [],
    "Ordinal" : []
}

# %%

# %%
type_records = []

for col in x_train.columns:
    pandas_dtype = str(x_train[col].dtype)
    
    
    if col in ['stops', 'class']:
        stat_type = 'Categorical (Ordinal)'
    elif col in ['airline', 'source_city', 'destination_city', 'departure_time', 'arrival_time', 'flight', 'source', 'destination', 'departure', 'arrival']:
        stat_type = 'Categorical (Nominal)'
    elif col in ['duration']:
        stat_type = 'Numerical (Continuous)'
    elif col in ['days_left', 'id']:
        stat_type = 'Numerical (Discrete)'
    else:
        stat_type = 'Numerical'
        
    type_records.append({
        'Feature': col,
        'Pandas Dtype': pandas_dtype,
        'Statistical Type': stat_type
    })

task2_df = pd.DataFrame(type_records)
print(task2_df.to_string(index=False))
# %%


x_train.describe()

# %%

x_train.isna().sum()
# %%



q1 = x_train['duration'].quantile(0.25)
q3 = x_train['duration'].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr 
upper_bound = q3 + 1.5 * iqr 
    

outliers = x_train[ (x_train['duration'] < lower_bound) | (x_train['duration'] > upper_bound) ]
print(f" found {len(outliers)} outliers in duration")



# %%

greater_than_10_unique = []

for col in x_train.columns:

    if x_train[col].dtype == "str":

        if(x_train[col].nunique() <= 10):
            print(col)
            print(x_train[col].unique())
            print()

        else:
            greater_than_10_unique.append(col)


print(greater_than_10_unique)

# %%


print(x_train['flight'].nunique())
# %%

# %%
# 'flight' has ~1500 unique codes -> one-hot would explode, and it mostly
# encodes airline + route which we already have. Drop it.
x_train = x_train.drop(columns=['flight'])
x_val = x_val.drop(columns=['flight'])

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

nominal_cols = ['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city']
ordinal_cols = ['stops', 'class']
numeric_cols = ['duration', 'days_left']

ct = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols),
    ('ord', OrdinalEncoder(categories=[['zero', 'one', 'two_or_more'], ['Economy', 'Business']]), ordinal_cols),
    ('ss', StandardScaler(), numeric_cols),
    ],
    verbose_feature_names_out=False,
).set_output(transform='pandas')

x_train_transform = ct.fit_transform(x_train)
x_val_transform = ct.transform(x_val)   # only transform on val -> no leakage

print(x_train_transform.shape)
x_train_transform.sample(4)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lin_reg = LinearRegression()
lin_reg.fit(x_train_transform, y_train)

print("train r2:", r2_score(y_train, lin_reg.predict(x_train_transform)))
print("val   r2:", r2_score(y_val, lin_reg.predict(x_val_transform)))

# %%
pd.Series(lin_reg.coef_, index=x_train_transform.columns).sort_values()
