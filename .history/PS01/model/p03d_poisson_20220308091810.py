import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import true
import util
from linearmodels import LinearModel


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
    
