# %%

import numpy as np
import pandas as pd

# %%
#SELECTION PART 2

df = pd.DataFrame(np.random.randn(9 , 4) , index = list("abcdefghi")  , columns = list("ABCD")) 
#or can simply use .range() fn for the index stuff 
# list(str) will split str into indv char arr

df.head()
# %%
print(df.shape)
print(df.columns)
# %%
df.info()
# %%
df.describe()
# %%
first_row = df.iloc[0]
first_row_loc = df.loc["a"]

# %%
print(first_row)
# %%
print(first_row_loc)
# %%
df.tail()
# %%
df.loc["a" , "A":"C"]
# %%
df.loc["a":"d" , :]
# %%
df = df.sort_values(by="A")
df.head()
# %%
df.iloc[0 : 4 , :]
# %%
