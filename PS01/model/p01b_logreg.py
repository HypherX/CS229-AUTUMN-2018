from linear_model import LinearModel
import util
import numpy as np


class LogisticRegression(LinearModel):
    """
    Logistic regression with Newton's Method as the solver.
    """

    def fit(self, x, y):
        """
        Run Newton's Method to minimize J(theta) for logistic regression
        :param x: Training example inputs. Shape: (m,n)
        :param y: Training example labels. Shape: (m,)
        """
        # Init theta
        m, n = x.shape
        self.theta = np.zeros(n)

        # Newton's method
        while True:
            # Save old theta
            theta_old = np.copy(self.theta)

            # Compute Hessian Matrix
            h_x = 1 / (1 + np.exp(np.dot(-x, self.theta)))
            H = (x.T * h_x * (1 - h_x)).dot(x) / m
            gradient_J_theta = x.T.dot(h_x - y) / m

            # Update theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)

            # End training
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break

    def predict(self, x):
        '''
        Make a prediction given new inputs x.
        :param x: Inputs of shape (m,n)
        :return: Outputs of shape (m,)
        '''
        return 1 / (1 + np.exp(-x.dot(self.theta)))


def main(train_path, eval_path, pred_path):
    '''
    Problem 1(b): Logistic regression with Newton's Method.
    :param train_path: Path to CSV file containing dataset for training
    :param eval_path: Path to CSV file containing dataset for evaluation
    :param pred_path: Path to save predictions
    '''
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train logistic regression
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, r'C:\Users\WIN10\Desktop\CS229\PS01\image\train_lr_dataset1.png')

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    y_pred = (y_pred > 0.5) + 0
    accuracy = sum(y_pred == y_eval) / len(y_eval)
    print(accuracy)
    # accuracy 0.9

    # Plot data and decision boundary
    util.plot(x_eval, y_eval, model.theta, r'C:\Users\WIN10\Desktop\CS229\PS01\image\valid_lr_dataset1.png')
    
    np.savetxt(pred_path, y_pred, fmt='%d')


if __name__ == '__main__':
    main('PS01\data\ds1_train.csv', 'PS01\data\ds1_valid.csv', r'C:\Users\WIN10\Desktop\CS229\PS01\predict\predict_lr_dataset1.txt')
