# %%

import enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.pylabtools import figsize
from jedi.third_party.typeshed.stubs.geopandas.geopandas.array import transform

# %%

#feature extraction

data = [
    {
        'age':4,
        'height': 96.0,
    },
    {
        'age':1,
        'height': 73.9,
    },
    {
        'age':3,
        'height': 88.9,
    },
    {
        'age':2,
        'height': 81.6,
    },

]


from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse = False)
data_transformed = dv.fit_transform(data)
print(data_transformed)
# %%
data_transformed.shape
# %%

#data imputation


from sklearn.impute import SimpleImputer

cols = ['age', 'sex', 'cp', 'trestbps', 'chol',  'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope','ca', 'thal', 'num']
heart_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',header=None, names=cols)

# %%

print(heart_data.columns)
# %%
print(heart_data.describe)
print(heart_data.info())
# %%

#check for missing

print(heart_data.isna().sum())
# %%

print(heart_data['ca'].unique())
print()
print(heart_data['thal'].unique())
print()
print((heart_data[['ca' , 'thal']] == '?').sum())
# %%
heart_data.replace('?' , np.nan , inplace=True)
# %%
imputer = SimpleImputer(missing_values= np.nan , strategy = 'mean')
imputer.set_output(transform="pandas")
heart_data_imputed = imputer.fit_transform(heart_data)
# %%

print(heart_data_imputed['ca'].unique())
print()
print(heart_data_imputed['thal'].unique())
# %%


cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
abalone_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', header=None, names=cols)
# %%

abalone_data.columns
# %%
print(abalone_data.info())
# %%
print(abalone_data['Sex'].unique())
# %%

abalone_data['Sex'] = abalone_data['Sex'].replace({"M" : 1 , "F" : 2 , "I" : 3}).astype(int)
print(abalone_data.info())
# %%
y = abalone_data.pop("Rings")
abalone_data.info()
# %%

abalone_data.describe().T
# %%


abalone_data.hist(figsize=(10 , 10))
plt.grid()
plt.show()
# %%

for col in abalone_data.columns:
    plt.figure()
    plt.hist(abalone_data[col])
    plt.grid()
    plt.title(col)
    plt.plot()


plt.show()
# %%

ax = abalone_data.plot.kde()
plt.grid()
plt.show()
# %%


import seaborn as sns

sns.histplot(abalone_data)
plt.show()

# %%

fig , axes = plt.subplots(3 , 3 , figsize = (12 , 10))
axes = axes.flatten()

for i , col in enumerate(abalone_data.columns):

    if i == 9:
        pass 
        
    sns.histplot(data = abalone_data , x=col , ax= axes[i])
    axes[i].set_title(col)
    axes[i].grid()
    
plt.tight_layout()
plt.show()
# %%
fig , axes = plt.subplots(3 , 3 , figsize = (12 , 10))
axes = axes.flatten()

for i , col in enumerate(abalone_data.columns):

    if i == 9:
        pass 
        
    sns.histplot(data = abalone_data , x=col , ax= axes[i] , kde=True)
    axes[i].set_title(col)
    axes[i].grid()
    
plt.tight_layout()
plt.show()
# %%

fig , axes = plt.subplots(3 , 3 , figsize=(12 , 10))
axes = axes.flatten()

for i , col in enumerate(abalone_data.columns):
    sns.boxplot(data = abalone_data , x = col , ax= axes[i] , orient="h")
    plt.grid()
    plt.title(col)


plt.tight_layout()
plt.show()
# %%

sns.boxplot(data = abalone_data , orient="h", palette="Set2")
plt.grid()
plt.show()
# %%

sns.violinplot(data = abalone_data , orient = "h" , palette= "Set2" , density_norm=  "width")
plt.grid()
plt.show()
# %%


x = np.array([4 , 2 , 5 , -2 , -100]).reshape(-1 ,1)
print(x)
print()
print(x.shape)
# %%

from sklearn.preprocessing import MaxAbsScaler

mas = MaxAbsScaler()
x_new = mas.fit_transform(x)
print(x_new)
# %%

from sklearn.preprocessing import MinMaxScaler

x = abalone_data
mm = MinMaxScaler().set_output(transform = "pandas")
X_normalized = mm.fit_transform(x)
# %%

X_normalized.hist(figsize=(12 , 10))
plt.tight_layout()
plt.show()

# %%


