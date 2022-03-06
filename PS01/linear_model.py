class LinearModel(object):
    """
    base class for linear model
    """

    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
        :param step_size: Step size for iterative solvers only
        :param max_iter: Maximum number of iteration for the solver
        :param eps: Threshold for determining convergence
        :param theta_0: Initial guess for theta. If None, use the zero vevtor
        :param verbose: Print loss values during training
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """
        Run solver to fit linear model
        :param X: Training example inputs. Shape: (m,n)
        :param y: Training example labels. Shape: (m,)
        """
        raise NotImplementedError('Subclass of LinearModel must implement fit method')

    def predict(self, x):
        """
        Make a prediction given new inputs x
        :param x: Inputs of shape (m,n)
        :return: Outputs of shape (m,)
        """
        raise NotImplementedError('Subclass of LinearModel must implement predict method')
