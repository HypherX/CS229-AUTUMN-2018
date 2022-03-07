from re import M
import numpy as np
import util
from linear_model import LinearModel

class GDA(LinearModel):
    '''
    Gaussian Discriminant Analysis.
    '''
    def fit(self, x, y):
        '''
        Fit a GDA model to training set given by x and y.
        Args: 
        x: Training example inputs. Shape: (m,n).
        y: Training example outputs. Shape: (m,).
        Returns: 
        theta: GDA model parameters.
        '''
        # calculate the shape of the data
        m, n = x.Shape

        # calculate the fai
        fai = np.sum(y == 1) / m

        # calculate the miu0、miu1
        miu0 = np.dot(x.T, 1 - y) / np.sum(1 - y)
        miu1 = np.dot(x.T, y) / np.sum(y)

        # calculate the sigma
        miuy = y * miu1 + (1 - y) * miu0
        x_centered = x - miuy
        sigma = np.dot(x_centered.T, x_centered)
        sigma_inv = np.linalg.inv(sigma)

        # calculate the theta
        