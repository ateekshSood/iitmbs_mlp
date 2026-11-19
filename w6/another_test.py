# %%

from pandas.core.common import random_state
from sklearn.datasets import load_wine

# %%

wine = load_wine(as_frame=True)

# %%

x , y = wine.data , wine.target 
# %%

from sklearn.model_selection import train_test_split, GridSearchCV

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.30 , random_state = 1)
# %%

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=10000 , C = 0.5 , random_state= 1 , solver='sag' )
log_reg.fit(x_train , y_train)


# %%

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test , log_reg.predict(x_test)))
# %%
log_reg.n_iter_


# %%


from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state= 1)

tree.fit(x_train , y_train)

params = {

    "criterion" : ["entropy" , "gini"],
    "splitter" : ["random" , "best"],
    "min_samples_leaf" : [ 2 , 4 , 6 , 8 , 10],
    "max_depth" : [3 , 4 , 5 , 6]
}

hp_tuning = GridSearchCV(tree , params , cv=4 )
hp_tuning.fit(x_train , y_train)

print(accuracy_score(y_test , hp_tuning.best_estimator_.predict(x_test)))
# %%
hp_tuning.best_params_
# %%


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(random_state=1)

params = {
    "n_estimators" : [100 , 500 , 1000],
    "learning_rate" : [0.5 , 1 , 2]
}

ada.fit(x_train , y_train)

hp_tuning = RandomizedSearchCV(ada , params , cv=4 )
hp_tuning.fit(x_train , y_train)

print(accuracy_score(y_test , hp_tuning.best_estimator_.predict(x_test)))
# %%

hp_tuning.best_params_
