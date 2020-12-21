import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('cleaned_data.csv')

print(df['make'].value_counts().count())

sns.pairplot(df)
plt.show()

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

a = df.describe()

print(df['car_type'].value_counts())
print(df['exterior_color'].value_counts())

from pandas.plotting import scatter_matrix
attributes = ["year", "model", "exterior_color", "interior_color", "number_of_keys", "price", "miles", "make","highway_mpg", "city_mpg", "car_type", "number_of_gears", "type_of_transmission", "number_of_doors"]
scatter_matrix(df[attributes], figsize=(12, 8))
plt.savefig('matrix.png')
plt.show()

df.drop(columns= ["interior_color", "model"], inplace= True)
df['number_of_gears'] = df['number_of_gears'].replace({"CVT": "0"})


df['price'] = df['price'].astype(int)
#creating dummies

df = pd.get_dummies(df, columns = ['exterior_color', 'type_of_transmission', 'car_type', 'make'], prefix = ['exterior_color', 'type_of_transmission', 'car_type', 'make'])
df
# df['price'] = df['price'].astype(int)
df['miles'] = df['miles'].str.replace('|', "")
df['miles'] = df['miles'].str.lstrip()
df[['price', 'number_of_keys', 'year', 'miles', 'highway_mpg', 'city_mpg', 'number_of_doors', 'number_of_gears']] = df[['price', 'number_of_keys', 'year', 'miles', 'highway_mpg', 'city_mpg', 'number_of_doors', 'number_of_gears']].astype(int)

X = df.iloc[:,1:]
y = df.iloc[:,0]


## important feature

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)

# plotting 7 most important features
important_feature = (model.feature_importances_)
feat_importance = pd.Series(important_feature, index = X.columns)
feat_importance.nlargest(8).plot(kind = 'barh')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 38)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


#HyperParameter
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


random_grid = {'n_estimators' : n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_random_forest = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator= rf_random_forest, param_distributions = random_grid, scoring= 'neg_mean_squared_error', n_iter=10, cv = 15, verbose= 2, random_state= 38, n_jobs=1)
rf_random.fit(X_train, y_train)

predictions = rf_random.predict(X_test)
predictions
sns.displot(y_test - predictions)
plt.show()

plt.scatter(y_test, predictions)
plt.show()

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

import pickle

file = open('random_forest_regression.pkl', 'wb')

pickle.dump(rf_random, file)