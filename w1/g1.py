# %%
from pathlib import Path

import numpy as np
import pandas as pd

# %%

# how many ? are present in the df

df = pd.read_csv("w1/Week1_GA_dataset.csv")

unknown_val = df == "?" # a df of False False True True per column 1 when column entry matches ? and 0 when it does not 
print(unknown_val.sum().sum())


df.replace("?" , np.nan , inplace=True)

# %%

#shape of the dataset

print(df.shape)

# %%

#value at 692 th index of the 0th index colum 


print(df.columns)
# %%
df['Date'].iloc[692]
# %%

#better 

df.iloc[692 , 0]


# %%

df.iloc[546 , 7]
# %%


print(df['Locality'].unique())
print(df['Locality'].dropna().unique())
# %%

print(df.isna().sum())
# %%


remaining_df = df[df.isna().sum(axis=1) <=2]
print(len(remaining_df))
# %%


remaining_df = df[df.isna().sum(axis=1) == 0]
print(len(remaining_df))
# %%


even_index_rows = df.iloc[0: len(df) :2  , :] 
print(even_index_rows.iloc[0 , 3])
# %%

even_index_rows.iloc[332 , 3]

# %%

odd_index_rows = df.iloc[1 : len(df) : 2 , :]
odd_index_rows.iloc[332 , 3]

# %%

odd_index_rows.iloc[100 , 5]

# %%

even_index_row_col = df.iloc[0: len(df) : 2 , 0 : len(df.columns)  :2]
even_index_row_col.iloc[255 ,3]
# %%

print(df.columns)
# %%
six_most_recent = pd.Series(df['Year'].unique()).sort_values(ascending=False).iloc[:6]

print(df['Year'].isin(six_most_recent).sum())
# %%

six_most_recent_better = df["Year"].drop_duplicates().nlargest(6)
print(df["Year"].isin(six_most_recent_better).sum())


# Sort by year index, grab the last 6, and sum their counts
print(df["Year"].value_counts().sort_index().tail(6).sum())

# %%

print(df.columns)
# %%
mask = (df['num_rooms'] == 3) & (df['num_bathrooms'] == 3)
df_yeah = df[mask]
print(df_yeah.shape)
print(mask.sum())

# %%
another_mask = (df['num_rooms'] == 3) | (df['num_bathrooms'] == 3)
print(another_mask.sum())
# %%
print(df.columns)

# %%
big_ass_mask = ((df['Year'] == 2022) & (df['Locality'] == "Greenwich") & (df['num_rooms'] ==3)) & ((df['Face'] == "North") | (df['Face'] == "East"))
print(big_ass_mask.sum())
# %%

print((pd.to_datetime(df["Date"]).dt.month == 8).sum())
# %%

df.groupby(df['Locality'])['Sale Price'].mean().sort_values(ascending=False)
