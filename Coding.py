#Shakir Mohammed, S17132995

# All imports/libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor 
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import time

# Using pandas to read the file in a csv format

data = pd.read_csv('GAMEDATASET.csv')

# Data exploration, features of dataset is explored for understanding
# 

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(data['Year_of_Release'],bins = 7)
plt.title('Release Distributation')
plt.xlabel('Year of Release')
plt.ylabel('Number of Games Released')
plt.show()

# This is where which Platform made the most sales globally

var = data.groupby('Platform').Global_Sales.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Platform')
ax1.set_ylabel('Sum Of Sales')
ax1.set_title("Platform wise Sum of Sales")
var.plot(kind='bar')

# The mean of Sales

expenditureNA = np.random.normal(data['NA_Sales'])
np.mean(expenditureNA)

expenditureJP = np.random.normal(data['JP_Sales'])
np.mean(expenditureJP)

expenditureEU = np.random.normal(data['EU_Sales'])
np.mean(expenditureEU)

expenditureOther = np.random.normal(data['Other_Sales'])
np.mean(expenditureOther)

expenditureGlobal = np.random.normal(data['Global_Sales'])
np.mean(expenditureGlobal)

# This is the median of all NA Sales

np.median(expenditureNA)

np.median(expenditureJP)

np.median(expenditureEU)

np.median(expenditureOther)

np.median(expenditureGlobal)

# This is the mode of all Na Sales

stats.mode(expenditureNA)

stats.mode(expenditureJP)

stats.mode(expenditureEU)

stats.mode(expenditureOther)

stats.mode(expenditureGlobal)

# Outliers, view on distribution of popular games

sns.boxplot(data['Global_Sales'],data['JP_Sales'])

sns.boxplot(data['Global_Sales'],data['NA_Sales'])

sns.boxplot(data['Global_Sales'],data['EU_Sales'])

sns.boxplot(data['Global_Sales'],data['Other_Sales'])

# Numerous scatter plots on sales against eachother for correlation

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data['Global_Sales'],data['EU_Sales'])


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data['JP_Sales'],data['EU_Sales'])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data['Other_Sales'],data['EU_Sales'])


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data['NA_Sales'],data['EU_Sales'])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data['Other_Sales'],data['EU_Sales'],data['NA_Sales'],data['JP_Sales'])

# Creating a new dataframe removing global sales under n (Size of reduction can be altered)

Global= data[data.Global_Sales >0.3]


# Removing features with no contribution.

del Global['Name']
del Global['NA_Sales']
del Global['JP_Sales']
del Global['Other_Sales']
del Global['Critic_Score']
del Global['Critic_Count']
del Global['User_Count']
del Global['Developer']
del Global['Rating']
del Global['User_Score']

# Normalization tool, transform catagorical values.

le = preprocessing.LabelEncoder()
Global.Platform = le.fit_transform(Global.Platform)
Global.Publisher =le.fit_transform(Global.Publisher.astype(str))
Global.Genre = le.fit_transform(Global.Genre.astype(str))

# Heat Map to show correlation of all features.

Global.corr()

corrMatrix = Global.corr()
print(corrMatrix)

sns.heatmap(corrMatrix, annot=True)
plt.show

# Pairplot to show correlation between all features

sns.pairplot(Global)

# Check for Nan

print(Global.isnull().sum().sum())

# Remove Nan's

Global.dropna(inplace= True)

# Seperate feaures and target
X = Global.iloc[:, :-1]
y = Global.iloc[:, -1]

# Split into training and test

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size = 0.2, random_state = 0)

# View shapes

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Linear Regression Model

Lr = linear_model.LinearRegression()
start = time.time()
Lr.fit(X_train, y_train)

stop = time.time()
print("Training time: ", (stop - start))

Lrpred = Lr.predict(X_test)

print('Coefficients: /n', Lr.coef_)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Lrpred))

print('Coefficiant of determination: %.2f'
     % r2_score(y_test, Lrpred))

# Scatter plot on prediction

plt.scatter( y_test,Lrpred, color='black')
plt.plot([0,30], [0,30], color='blue')
plt.plot(y_test,pred,linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show

# Decision Trees 

Dtc = DecisionTreeRegressor()

start = time.time()
Dtc.fit(X_train,y_train)
stop = time.time()
print("Training time: ", (stop - start))

Dtcpred = Dtc.predict(X_test)


print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Dtcpred))

print('Coefficiant of determination: %.2f'
     % r2_score(y_test, Dtcpred))

# Scatter Plot of prediction 

plt.figure()
plt.scatter(y_test,Dtcpred, s=20, edgecolor="black",
            c="darkorange", label="data")

plt.xlabel("Data")
plt.ylabel("Target")
plt.title("Prediction Graph")
plt.legend()
plt.show()

# Random forrest 

Rfr = RandomForestRegressor()

start = time.time()
Rfr.fit(X_train,y_train)
stop = time.time()
print("Training time: ", (stop - start))

Rfrpred = Rfr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Rfrpred))

print('Coefficiant of determination: %.2f'
     % r2_score(y_test, Rfrpred))

# Gradient boosting 0.52

Gbr = GradientBoostingRegressor()
start = time.time()
Gbr.fit(X_train,y_train)
stop = time.time()
print("Training time: ", (stop - start))
Gbrpred = Gbr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Gbrpred))

print('Coefficiant of determination: %.2f'
     % r2_score(y_test, Gbrpred))

# Svr

Svr = SVR()
start = time.time()
Svr.fit(X_train, y_train)
stop = time.time()
print("Training time: ", (stop - start))
Svrpred = Svr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Svrpred))

print('Coefficiant of determination: %.2f'
     % r2_score(y_test, Svrpred))

# Cross validation preparation 

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

# GridSearchCV

rfr = RandomForestRegressor()

# All parameters currently in use
print('Parameters currently in use:\n')
print(rfr.get_params())

# Trees in a forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Amount of features to observe during every split

max_features = ['auto', 'sqrt']

# Topmost level in tree

max_depth = [5,9,12]
max_depth.append(None)

# Minimal integer of samples for split

min_samples_split = [2, 5, 10]

# Minimum number of samples at each leaf node

min_samples_leaf = [1, 2, 4]

# Procedure of selecting samples for training every individual tree

bootstrap = [True, False]

# Parameter grid

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Print paramters 

print(param_grid)

# Use the param grid for a search of best hyperparameters
# Model to tune
rf = RandomForestRegressor()

# Fortuitous search of parameters (3 Fold Cross Validation)
# Cast around across different combinations, and use all available 

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, n_iter = 100, cv = cv, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

start = time.time()
rf_random.fit(X_train, y_train)
stop = time.time()
print("Training time: ", (stop - start))

# find best parameters

rf_random.best_params_

# Random forest with best parameters

rf = RandomForestRegressor(n_estimators=1800, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="auto", bootstrap=True)

start = time.time()

rf.fit(X_train, y_train)

stop =time.time()
print("Training time: ", (stop - start))
rfpred = rf.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, rfpred))

print(metrics.mean_absolute_error(y_test, rfpred))

print(np.sqrt(metrics.mean_squared_error(y_test, rfpred)))

print('Coefficiant of determination: %.2f'
     % r2_score(y_test, rfpred))

# LinearRegression search for best parameters

regrnone = linear_model.LinearRegression()

start = time.time()
regrnone.fit(X_train, y_train)
stop = time.time()
print("Training time: ", (stop - start))

prednone = regrnone.predict(X_test)

print('Coefficients: /n', regr.coef_)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, prednone))

print('Coefficiant of determination: %.2f'
     % r2_score(y_test, pred))

cv_results = cross_val_score(regrnone, X, y, cv=4,
                             scoring="neg_mean_squared_error")

# Finding parameters for Linear Regression

lr = linear_model.LinearRegression()

# Parameters used currently

print('Parameters currently in use:\n')
print(lr.get_params())

fit_intercept = [True, False]

copy_X = [True, False]

n_jobs = [1,-1,-2]

normalize = [True, False]

# Parameter grid

param_grid2 = {'fit_intercept': fit_intercept,
               'copy_X': copy_X,
               'n_jobs': n_jobs,
               'normalize': normalize}

pprint(param_grid2)

# Search using 3 fold cross validation

lr_random = RandomizedSearchCV(estimator = lr, param_distributions = param_grid2, n_iter = 100, cv = cv, verbose=2, random_state=42, n_jobs = -1)

# Fit 
start = time.time()
lr_random.fit(X_train, y_train)
stop = time.time()
print("Training time: ", (stop - start))

# Find the best parameters

lr_random.best_params_

# Linear regression with best parameters

lr = linear_model.LinearRegression(normalize=True, n_jobs=1, fit_intercept=True, copy_X=True)

start = time.time()
lr.fit(X_train, y_train)
stop = time.time()
print("Training time: ", (stop - start))

predlr = lr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, predlr))

print('Coefficiant of determination: %.2f'
     % r2_score(y_test, predlr))






































