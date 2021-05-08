import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg.linalg import LinAlgError
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class LinearRegressionAnalytics:
    def __init__(self):
        self.betas = None
        self.fitted = False

    @staticmethod
    def preprocess_x(X):
        n = len(X)
        X = np.hstack([np.ones((n, 1)), X])
        return X

    def fit(self, X, y):
        """
        This function estimates betas based on analytic approach
        betas = (X.T @ X) ^ -1 @ X.T @ y
        :param X: (n, k)
        :param y: (n, 1)
        """
        X = self.preprocess_x(X)
        try:
            self.betas = np.linalg.inv(X.T @ X) @ X.T @ y
            self.fitted = True
        except LinAlgError("Non Invertible Matrix") as e:
            print(e)
        return self

    def predict(self, X):
        X = self.preprocess_x(X)
        y_hat = X @ self.betas
        return y_hat

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean((y - y_hat) ** 2)
class Model:
    """
     This class implements linear regression computing OLS coefs.
    """

    def __init__(self):
        self.fitted = False
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimates params based on this function
        params = (X.T @ X) ^ -1 @ X.T @ y
        :param X: (n, k)
        :param y: (n, 1)
        """
        # X = np.hstack([np.ones(len(X))[:, np.newaxis], X])
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        try:
            self.params = np.linalg.inv(X.T @ X) @ X.T @ y
            self.fitted = True
        except LinAlgError("Non Invertible Matrix") as e:
            print(e)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        y_hat = X @ self.params
        return y_hat

    def plot_predictions(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray = None):
        """
        Plots the line and the actual points.
        :param X: features
        :param y: true value
        :param y_hat: predicted value
        :return:
        """
        if y_hat is None:
            y_hat = self.predict(X)

        plt.scatter(X, y, c='blue', label="Observed values.")
        plt.legend()
        plt.plot(X, y_hat, c="r", label="Predicted values")
        plt.legend()

        plt.title(f"Fitted Regression. intercept_{self.params[0][0]} coefs_{self.params[1:]}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean((y - y_hat) ** 2)

    def save(self):
        pass

    def load(self):
        pass


lr = LinearRegression()
# egressor = LinearRegression()
PATH = r'C:\Users\DELL\Desktop\Python\ML - ACA\First model\Car details v3.csv'
Columns = ['year', 'selling_price', 'km_driven',
           'mileage', 'engine', 'max_power']

# Preparing the data
df = pd.read_csv(PATH)

# dropping some columns, because they are too hard to get info from
df.drop('fuel', axis=1, inplace=True)
df.drop('torque', axis=1, inplace=True)
df.drop('transmission', axis=1, inplace=True)
df.drop('seats', axis=1, inplace=True)
# dropping some columns, but maybe later I will return them
df.drop('seller_type', axis=1, inplace=True)
df.drop('name', axis=1, inplace=True)
df.drop('owner', axis=1, inplace=True)

# year
df['year'] -= 1982
df['year'].fillna(np.mean(df['year']))

# selling_price
df['selling_price'] /= 1000
df['selling_price'].fillna(np.mean(df['selling_price']))
# km_driven
df['km_driven'] /= 1000
df['km_driven'].fillna(np.mean(df['km_driven']))


def what_I_want(str):
    df[str] = df[str][pd.notna(df[str])].str.split()
    df[str] = df[str].str[0]
    df[str] = pd.to_numeric(df[str])
    df[str] = df[str].fillna(np.mean(df[str][pd.notna(df[str])]))

# mileage
what_I_want('mileage')

# engine
what_I_want('engine')

# max_power
what_I_want('max_power')

# making train and test data
y = df['selling_price']
df.drop('selling_price', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

lr.fit(X_train, y_train)
print(X_test.head(), y_test.head())
example = pd.DataFrame(data=[[33,      110.0,    14.95,  2489.0,      93.70]], columns=['year', 'km_driven', 'mileage', 'engine', 'max_power'])
print(lr.predict(example))

print(lr.score(X_train, y_train))
