import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PolynomailRegression():

    def __init__(self, degree, learning_rate, iterations):

        self.degree = degree

        self.learning_rate = learning_rate

        self.iterations = iterations

    # function to transform X

    def transform(self, X):

        # initialize X_transform

        X_transform = np.ones((self.m, 1))

        j = 0

        for j in range(self.degree + 1):

            if j != 0:
                x_pow = np.power(X, j)

                # append x_pow to X_transform

                X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis=1)

        return X_transform

        # function to normalize X_transform

    def normalize(self, X):

        X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)

        return X

    # model training

    def fit(self, X, Y):

        self.X = X

        self.Y = Y

        self.m, self.n = self.X.shape

        # weight initialization

        self.W = np.zeros(self.degree + 1)

        # transform X for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n

        X_transform = self.transform(self.X)

        # normalize X_transform

        X_normalize = self.normalize(X_transform)

        # gradient descent learning

        for i in range(self.iterations):
            h = self.predict(self.X)

            error = h - self.Y

            # update weights

            self.W = self.W - self.learning_rate * (1 / self.m) * np.dot(X_normalize.T, error)

        return self

    # predict

    def predict(self, X):

        # transform X for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n

        X_transform = self.transform(X)

        X_normalize = self.normalize(X_transform)

        return np.dot(X_transform, self.W)


# def esti_cof(x,y):
#     n = np.size(x)
#
#     Mx = np.mean(x)
#     My = np.mean(y)
#
#     SsXy = np.sum(y*x) - n*My*Mx
#     SsXx = np.sum(y * x) - n * My * Mx
#
#     b_1= SsXy / SsXx
#     b_0 = My -b_1*Mx
#     return(b_0,b_1)
#
#
# def plot_regression_line(x, y, b):
#     # plotting the actual points as scatter plot
#
#     plt.scatter(x, y, color="m",
#                 marker="o", s=30)
#
#     # predicted response vector
#     y_pred = b[0] + b[1] * x
#
#     # plotting the regression line
#     plt.plot(x, y_pred, color="g")
#
#     # putting labels
#     plt.xlabel('x')
#     plt.ylabel('y')
#
#     # function to show plot
#     plt.show()

def main():
    datas = pd.read_csv("Task1 - dataset - pol_regression.csv")
    x = datas.iloc[:, 1:2].values
    y = datas.iloc[:, 2].values

    print(x)
    model = PolynomailRegression(degree=10, learning_rate=0.1, iterations=500)
    model.fit(x, y)
    y_pred = model.predict(y)
    plt.scatter(x, y, color='blue')

    plt.plot(x, y_pred, color='orange')

    plt.title('X vs Y')

    plt.xlabel('X')

    plt.ylabel('Y')

    plt.show()

     # b = esti_cof(x,y)
    # # print("Estimated coefficients:\nb_0 = {}  \
    # #           \nb_1 = {}".format(b[0], b[1]))
    # # plot_regression_line(x, y, b)


main()
