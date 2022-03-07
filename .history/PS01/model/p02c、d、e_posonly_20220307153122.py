from re import L
import numpy as np
from torch import mode
import util
from p01b_logreg import LogisticRegression


def main(train_path, valid_path, test_path, pred_path):
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
    x_train, t_train = util.load_csv(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_csv(test_path, label_col='t', add_intercept=True)

    model_c = LogisticRegression()
    model_c.fit(x_train, t_train)

    util.plot(x_test, t_test, model_c.theta, )