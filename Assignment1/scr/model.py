# import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
# %matplotlib inline
# -------------------------------

# Display settings on console and sub_plotting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 320)
print("\nStart training/testing")
# ---------------------------------------------------

# read train databases
train_data = pd.read_csv('../train_set/train_o.csv')
x_train = train_data.drop(columns=['Delay'], axis=1)
y_train = train_data['Delay']

test_data = pd.read_csv('../test_set/test.csv')
x_test = test_data.drop(columns=['Delay'], axis=1)
y_test = test_data['Delay']
# ---------------------------------------------------

# Implementation of linear regression model
X_train, x_val, Y_train, y_val = train_test_split(x_train, y_train,
                                                  train_size=0.7, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(f"\nModel intercept : {regressor.intercept_}")
print(f"Model coefficient : {regressor.coef_}")
y_pred = regressor.predict(x_val)
print('\nMSE for train Dataset:', mean_squared_error(y_val, y_pred))
y_pred = regressor.predict(x_test)
print('MSE for test Dataset:', mean_squared_error(y_test, y_pred))
print('R2_score for the model:', r2_score(y_test, y_pred))
# ---------------------------------------------------

# Using Lasso
lasso = Lasso()
lasso.fit(X_train, Y_train)
print(f"\nModel coefficient : {lasso.coef_}")
y_pred = lasso.predict(x_val)
print('\nMSE for train Dataset:', mean_squared_error(y_val, y_pred))
y_pred = lasso.predict(x_test)
print('MSE for test Dataset:', mean_squared_error(y_test, y_pred))
print('R2_score for the model:', r2_score(y_test, y_pred))

# Tuning alpha
X_train, x_val, Y_train, y_val = train_test_split(x_train, y_train,
                                                  train_size=0.7, random_state=0)
alphas = np.linspace(0.015, 0.0001, 20)
losses = []
for alpha in alphas:
    lasso = Lasso(alpha)
    lasso.fit(X_train, Y_train)
    y_pred = lasso.predict(x_val)
    mse = mean_squared_error(y_pred, y_val)
    losses.append(mse)

plt.plot(alphas, losses)
plt.title("Lasso alpha value selection")
plt.xlabel("alpha")
plt.ylabel("Mean squared error")
plt.show()

best_alpha = alphas[np.argmin(losses)]
print("\nBest value of alpha:", best_alpha)

lasso = Lasso(best_alpha)
lasso.fit(x_train, y_train)
y_pred = lasso.predict(x_test)
print("\nMSE on testset:", mean_squared_error(y_test, y_pred))
print('R2_score for the model:', r2_score(y_test, y_pred))
# ---------------------------------------------------

# Polynomial regression
X = x_train["Flight duration minute"].values
y = y_train.values

degrees = [1, 5, 24]

plt.figure(figsize=(20, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i])
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(X.min(), X.max(), 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]),'r', label="Model")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel('Flight duration minute')
    plt.ylabel('Delay')
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i],
              -scores.mean(), scores.std()))
plt.show()
polynomial_features = PolynomialFeatures(degree=5)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])

pipeline.fit(X[:, np.newaxis], y)
y_pred = pipeline.predict(X[:, np.newaxis])
print('\nMSE for train Dataset:', mean_squared_error(y, y_pred))
X = x_test["Flight duration minute"].values
y = y_test.values
y_pred = pipeline.predict(X[:, np.newaxis])
print("MSE on testset:", mean_squared_error(y, y_pred))
print('R2_score for the model:', r2_score(y_test, y_pred))
print('\nFinish')
# ---------------------------------------------------
