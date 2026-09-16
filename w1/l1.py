# %%
import pandas as pd
import numpy as np
# %%

#check version 

print(pd.__version__)
# %%

from sklearn.datasets import load_diabetes 

diabetes = load_diabetes(as_frame = True)
 # without this it will  load bunch
print(type(diabetes['data']))
# %%

df = diabetes['data']
df.head(4)
# %%
#creating series 


cities = pd.Series(['Mumabi' , 'Banglore' , 'Chennai' ,"yeh"])
population = pd.Series([10000 , 100000 , 10000])

df = pd.DataFrame( {"City" : cities , "Populaiton" : population} )
df
df.fillna(df['Populaiton'].mean())
# %%

