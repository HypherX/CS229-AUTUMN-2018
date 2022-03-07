import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import numpy as np
import util
from p01b_logreg import LogisticRegression


def main_c(train_path, valid_path, test_path, pred_path):
    '''
    Problem 2: Logistic regression for incomplete, positive-only labels
    Run under the following conditions:
    1. on y-labels
    2. on l-labels
    3. on l-labels with correction factor alpha
    Args:
    train_path: Path to CSV file containing training set
    valid_path: Path to CSV file containing validation set
    test_path: Path to CSV file containing test set
    pred_path: Path to save predictions
    '''
    # Problem c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model_c = LogisticRegression()
    model_c.fit(x_train, t_train)

    util.plot(x_test, t_test, model_c.theta, r'C:\Users\WIN10\Desktop\CS229\PS01\image\problem2-(c).png')

    pred_c = model_c.predict(x_test)
    np.savetxt(pred_path, pred_c > 0.5, fmt='%d')


def main_d(train_path, valid_path, test_path, pred_path):
    '''
    Problem 2: Logistic regression for incomplete, positive-only labels
    Run under the following conditions:
    1. on y-labels
    2. on l-labels
    3. on l-labels with correction factor alpha
    Args:
    train_path: Path to CSV file containing training set
    valid_path: Path to CSV file containing validation set
    test_path: Path to CSV file containing test set
    pred_path: Path to save predictions
    '''
    # Problem c
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    model_d = LogisticRegression()
    model_d.fit(x_train, y_train)

    util.plot(x_test, y_test, model_d.theta, r'C:\Users\WIN10\Desktop\CS229\PS01\image\problem2-(d).png')

    pred_d = model_d.predict(x_test)
    np.savetxt(pred_path, pred_d > 0.5, fmt='%d')


def main_e(train_path, valid_path, test_path, pred_path):
    '''
    Problem 2: Logistic regression for incomplete, positive-only labels
    Run under the following conditions:
    1. on y-labels
    2. on l-labels
    3. on l-labels with correction factor alpha
    Args:
    train_path: Path to CSV file containing training set
    valid_path: Path to CSV file containing validation set
    test_path: Path to CSV file containing test set
    pred_path: Path to save predictions
    '''
    # Problem c
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    model_e = LogisticRegression()
    model_e.fit(x_train, y_train)

    alpha = np.mean(model_e.predict(x_valid))
    correction = 

    util.plot(x_test, y_test, model_d.theta, r'C:\Users\WIN10\Desktop\CS229\PS01\image\problem2-(d).png')

    pred_d = model_d.predict(x_test)
    np.savetxt(pred_path, pred_d > 0.5, fmt='%d')


if __name__ == '__main__':
    main_c('PS01\data\ds3_train.csv', 'PS01\data\ds3_valid.csv', 'PS01\data\ds3_test.csv', r'C:\Users\WIN10\Desktop\CS229\PS01\predict\problem2-(c).txt')
    main_d('PS01\data\ds3_train.csv', 'PS01\data\ds3_valid.csv', 'PS01\data\ds3_test.csv', r'C:\Users\WIN10\Desktop\CS229\PS01\predict\problem2-(d).txt')