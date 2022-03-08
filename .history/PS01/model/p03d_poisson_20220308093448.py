import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import true
import util
from linear_model import LinearModel


class PoissonRegression(LinearModel):
    def fit(self, x, y):
        m, n = x.shape
        self.theta = np.zeros(n)

        # optimization
        while true:
            # update the theta
            theta = np.copy(self.theta)
            self.theta += self.step_size * (x.T.dot(y - np.exp(x.dot(self.theta)))) / m

            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break
    
    def predict(self, x):
        # prediction
        return np.exp(np.dot(x, self.theta))


def main(train_path, eval_path, pred_path):
    # Load the dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Build and fit model
    model = PoissonRegression()
    model.fit(x_train, y_train)

    # Prediction
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    # Plot data and decision boundary
    np.savetxt(pred_path, y_pred)
    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03d.png')


if __name__ == '__main__':
    main('PS01\data\ds4_train.csv', 'PS01\data\ds4_valid.csv', r'C:\Users\WIN10\Desktop\CS229\PS01\predict\problem3-(d).txt')