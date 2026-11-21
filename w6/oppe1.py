# %%

import pandas as pd

# %%

df = pd.read_csv("w6/NPPE1_Preprocessing1.csv")
# %%

df.info()
# %%

df.describe()
# %%

print( (df['RM'] >= 5).sum()) 
# %%
avg_top_10 = df['PRICE'].nlargest(10).mean()
print(avg_top_10)
# %%

print(df['RM'].isna().sum())
# %%
print(df['RM'].unique())
# %%
print(len(df[df['RM'] == -1]))
print((df['RM'] == -1).sum())
# %%

df.columns
# %%
print(df['AGE'].isna().sum())
# %%
print(df['AGE'].nunique())
# %%
print(df['AGE'].unique())
print( (df['AGE'] < 0).sum() )
# %%

print(df['RIVERSIDE'].unique())
# %%

print( (df['RIVERSIDE'] == 'UNKNOWN').sum())
# %%

print( ((df['RIVERSIDE'] == 'YES') & (df['AGE'] < 50)).sum() )
# %%
df['HIGHWAYCOUNT'].head(3)
# %%

df.head(5)
# %%


print( ((df['HIGHWAYCOUNT'] == 6) | (df['HIGHWAYCOUNT'] == 7) | (df['HIGHWAYCOUNT'] == 8)).sum() )
print(df['HIGHWAYCOUNT'].isin([6 , 7 , 8]).sum())
# %%

df['PRICE'].head(3)
# %%
import numpy as np

# %%


bins = [-np.inf , 10 , 20 , 30 , 40 , np.inf]
labels = ['Category1' , 'Category2' , 'Category3' , 'Category4' , 'Category5']


df['CATEGORY'] = pd.cut(df['PRICE'] , bins = bins , labels = labels , right = False)

# %%
df['CATEGORY'].value_counts()
# %%
print(df['CATEGORY'].idxmax())


# %%
df.columns
# %%

from sklearn.model_selection import train_test_split

y = df['PRICE']
x = df.drop(columns=['PRICE'])

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.30 , random_state = 0)

# 
# cat_si = SimpleImputer(missing_values= 'UNKNOWN' ,  strategy = 'constant' , fill_value = np.nan)
# num_si = SimpleImputer(missing_values= -1 ,  strategy = 'constant' , fill_value = np.nan)
# num_si = SimpleImputer(missing_values= -1 ,  strategy = 'constant' , fill_value = np.nan)
# 
# 
# ct = ColumnTransformer(
#     [
#         ( 'nan_value_replace' , si , ['RM' , 'RIVERSIDE' , 'AGE'])
#     ],
#     remainder="passthrough"
# )
# 
# 
# x_train_processed = ct.fit_transform(x_train)


x_train['RIVERSIDE'] = x_train['RIVERSIDE'].replace('UNKNOWN' , np.nan)
x_train['RM'] = x_train['RM'].replace(-1 , np.nan)
x_train['AGE'] = x_train['AGE'].replace(-2 , np.nan)
# %%


# %%


print(len(x_train))
print(len(y_train))
# %%


x_train.columns
x_train = x_train.drop(columns=['CATEGORY'])
# %%
x_train.columns
# %%


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

rm_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    MinMaxScaler()
)

age_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler()
)

riverside_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(sparse_output=False)
)

    
ct = ColumnTransformer([
    ('min_max_sacler1' , MinMaxScaler() , ['CRIM' , 'ZN']),
    ('standard_scaler' , StandardScaler() ,  ['INDUS'] ),
    ('min_max_sacler2' , MinMaxScaler() , ['POLINDEX' , 'DIS' , 'HIGHWAYCOUNT' , 'TAX' , 'PTRATIO' , 'IMM' , 'BPL']),
    ('mean_imputer' ,rm_pipeline , ['RM']),
    ('median_imputer' , age_pipeline , ['AGE']),
    ('most_freq_imputer' , riverside_pipeline , ['RIVERSIDE']),

] , remainder="passthrough")

ct.set_output(transform="pandas")

x_train_preprocessed = ct.fit_transform(x_train)
# %%


len(x_train_preprocessed.columns)


# %%

x_test['RIVERSIDE'] = x_test['RIVERSIDE'].replace('UNKNOWN' , np.nan)
x_test['RM'] = x_test['RM'].replace(-1 , np.nan)
x_test['AGE'] = x_test['AGE'].replace(-2 , np.nan)

x_test_trasnformed = ct.transform(x_test)

print(x_test_trasnformed.mean().mean())
