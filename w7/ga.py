# %%
from sklearn.datasets import fetch_california_housing

# %%

data = fetch_california_housing(as_frame = True)
x  , y = data.data , data.target

# %%
from sklearn.model_selection import train_test_split

x_train , x_test , y_train  , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 1)
# %%


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score , root_mean_squared_error

# %%


lin_reg = LinearRegression()
lin_reg.fit(x_train , y_train)

print(r2_score(y_test , lin_reg.predict(x_test)))
print(root_mean_squared_error(y_test , lin_reg.predict(x_test)))


# %%
ada_reg = AdaBoostRegressor()
ada_reg.fit(x_train , y_train)

print(r2_score(y_test , ada_reg.predict(x_test)))
print(root_mean_squared_error(y_test , ada_reg.predict(x_test)))

# %%

tree = DecisionTreeRegressor()
tree.fit(x_train , y_train)

print(r2_score(y_test , tree.predict(x_test)))
print(root_mean_squared_error(y_test , tree.predict(x_test)))

# %%
from sklearn.datasets import load_iris

data = load_iris(as_frame = True)

x , y = data.data , data.target

x_train , x_test , y_train , y_test = train_test_split(x ,  y , test_size =0.33 , random_state = 1)

# %%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=1)

log_reg.fit(x_train , y_train)


# %%

from sklearn.metrics import confusion_matrix , classification_report

print(confusion_matrix(y_test , log_reg.predict(x_test)))
print(classification_report(y_test , log_reg.predict(x_test)))
# %%

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

x , y = fetch_20newsgroups(subset="train" , return_X_y = True)

vectorizer = TfidfVectorizer()
x_tfidf = vectorizer.fit_transform(x)

print(x_tfidf.shape)
