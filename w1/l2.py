# %%
from numpy.matlib import percentile
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

df = load_diabetes(as_frame=True)
# %%
y = df['target']
df = df['data']
# %%

print(y)
y = np.array(y) # pd series to numpy array
type(y)
# %%
df.head()
# %%

df["bmi"].describe()
# %%
bmi_ge_eq_mea_mask = df["bmi"] >= df["bmi"].mean()
df_bmi_ge_eq_mean = df[bmi_ge_eq_mea_mask]
df_bmi_ge_eq_mean.head()
# %%

# examine some entries of the df 

df.head()
# %%
? df.head # to know about any fn
# %%

print(df.tail())
print(len(df))
# %%



# to know about the shape of the df 
print(df.shape)
# %%
#how to find the col name 

print(df.columns)
# %%
df.info()
# df["age"] = df["age"].astype(int)
# %%
df.info()
# %%
print(df["age"].describe())
print(df["age"].head())
# %%
print(df.describe(percentiles = [0.2 , 0.6 , 0.8]).T)
# use .T for better view
