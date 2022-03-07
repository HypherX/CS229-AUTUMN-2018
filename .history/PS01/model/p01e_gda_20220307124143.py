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
        m, n = x.shape

        # calculate the fai
        fai = np.sum(y == 1) / m

        # calculate the miu0、miu1
        miu0 = np.dot(x.T, 1 - y) / np.sum(1 - y)
        miu1 = np.dot(x.T, y) / np.sum(y)

        y = y.reshape(m, -1)

        # calculate the sigma
        miuy = y * miu1 + (1 - y) * miu0
        x_centered = x - miuy
        sigma = np.dot(x_centered.T, x_centered)
        sigma_inv = np.linalg.inv(sigma)

        # calculate the theta and theta_0
        theta = np.dot(sigma_inv, miu1 - miu0)
        theta_0 = miu0.T @ sigma_inv @ miu0 - miu1.T @ sigma_inv @ miu1 - np.log(1 - fai / fai)

        self.theta = np.insert(theta, 0, theta_0)

    def predict(self, x):
        '''
        Make a prediction given the new inputs x.
        param x: Inputs of shape (m, n).
        return: Outputs of shape (m,).
        '''
        return util.add_intercept(x) @ self.theta >= 0
    

def main(train_path, eval_path, pred_path):
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    
    # Train GDA
    model = GDA()
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, r'C:\Users\WIN10\Desktop\CS229\PS01\image\train_gda.png')

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_eval)
    y_pred = (y_pred > 0.5) + 0
    accuracy = sum(y_pred == y_eval) / len(y_eval)
    print(accuracy)
    # accuracy 0.9

    # Plot data and decision boundary
    util.plot(x_eval, y_eval, model.theta, r'C:\Users\WIN10\Dekstop\CS229\PS01\image\valid_gda.png')

    np.savetxt(pred_path, y_pred, fmt='%d')


if __name__ == '__main__':
    main('PS01\data\ds1_train.csv', 'PS01\data\ds1_valid.csv', r'C:\Users\WIN10\Desktop\CS229\PS01\predict\predict_gda.txt')