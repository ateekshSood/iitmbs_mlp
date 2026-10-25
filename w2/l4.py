# %%

# data preprocessing


# %%

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# %%

housing = fetch_california_housing(as_frame=True)
df = housing.frame.iloc[:500, :].copy()

np.random.seed(42)
mask_medinc = np.random.rand(len(df)) < 0.10
mask_age = np.random.rand(len(df)) < 0.10
df.loc[mask_medinc, 'MedInc'] = np.nan
df.loc[mask_age, 'HouseAge'] = np.nan

df.loc[10, 'Population'] = 50000.0
df.loc[25, 'Population'] = 75000.0

df.info()
# %%
print(df['MedInc'].isna().sum())
print(df['MedInc'].head(5))
# %%

from sklearn.impute import SimpleImputer

simple_imp = SimpleImputer(strategy='median')


df[['MedInc']] = simple_imp.fit_transform(df[['MedInc']])
# %%

print(df['MedInc'].isna().sum())
# %%
print(df.columns)
# %%
df[['HouseAge']] = simple_imp.fit_transform(df[['HouseAge']])
# %%
print(df['HouseAge'].isna().sum())
# %%

