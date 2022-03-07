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
        fai = sum(y == 1) / m

        # calculate the miu0
        miu0 = 