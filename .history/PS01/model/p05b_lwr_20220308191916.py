from turtle import Turtle
import matplotlib.pyplot as plt
import numpy as np
import util 
from linear_model import LinearModel


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None
        self.theta = None
    
    def fit(self, x, y):
        # Calculate the W
        m, n = x.shape
        W = np.zeros((m, m))
        for i in range(m):
            W[i, i] = np.exp(-np.sum((x - x[i]) ** 2, axis=1) / (2 * self.tau ** 2))
        
        # Calculate the theta
        self.theta = np.linalg.inv(x.T.dot(W).dot(x)).dot(x.T).dot(W).dot(y)

    def predict(self, x):
        m, n = x.shape
        y_pred = np.zeros(m)

        # Prediction
        y_pred = self.theta.T.dot(x)
        
        return y_pred


def main(tau, train_path, eval_path):
    # Load the data
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Build the model and predict
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    # Calculate the mse loss
    mse = np.mean((y_pred - y_eval) ** 2)
    print(mse)

    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_eval, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(r'C:\Users\WIN10\Desktop\CS229\PS01\image\problem5-(b).png')