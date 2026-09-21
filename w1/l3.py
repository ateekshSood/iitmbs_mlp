# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# %%
X : pd.DataFrame
y : pd.Series
X  , y  = load_diabetes(as_frame=True , return_X_y = True)
# %%
print(type(y))
print(y.head(3))
print(y.describe())
print(y.info())
# %%
df = X
# %%
df.head()
# %%
df.info()
# %%
df.describe()
# %%
 #DATA SELECTION

df.columns
# %%
df['age']
# %%
print(df['age'][0])
# %%
df['age'][:5]
# %%
df['age'][-5:]
# %%
df['age'][100:200]
# %%

#give a list of columns to select em 
df[['age' , 'sex']]
# %%
df[['age' , 'sex']][:5]
# %%
df[['age' , 'sex']][-5:]
# %%
df[['age' , 'sex']].iloc[1]
# %%

# .loc = location 
# .iloc = integer location 
# 
# in iloc we can give posi in term of integer value 

# %%

df.iloc[0]
# %%
df.loc[0]
# %%
# right nwo they giivng same output cuz the index is same . iloc will only work with integer index but if we changed the index 
# to some str we have to use loc that will work not iloc.

# %%
print(df.iloc[0]['age'])
print(df.iloc[0 , 0])
print(df.iloc[0 , 0:2])
# %%
print(df.loc[0 , 'age'])
# %%
print(df.loc[0 , ['age' , 'sex']])
# %%
print(df.iloc[0 , [1,5,7,9]])
# %%

# CONDITIONAL SLICING 

#all rows where age is less than this 
rows = df.loc[df.age < 5.383060e-03]
print(type(rows))
print(rows.head())
# %%
print(rows.iloc[0])
