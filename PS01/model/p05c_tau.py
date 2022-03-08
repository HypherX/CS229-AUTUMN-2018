import matplotlib.pyplot as plt
import numpy as np
import util
from p05b_lwr import LocallyWeightedLinearRegression

def main(tau_list, train_path, valid_path, test_path, pred_path):
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    model = LocallyWeightedLinearRegression(tau=0.5)
    model.fit(x_train, y_train)

    mse_list = []
    for tau in tau_list:
        model.tau = tau
        y_pred = model.predict(x_eval)

        mse = np.mean((y_pred - y_eval) ** 2)
        mse_list.append(mse)
        print(f'Validation set: tau={tau}, MSE={mse}')

        plt.figure()
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_eval, y_eval, 'ro', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(fr'C:\Users\WIN10\Desktop\CS229\PS01\image\problem5-(c{tau}).png')
    
    tau_best = tau_list[np.argmin(mse_list)]
    print(f'Best tau={tau_best}, MSE_low={min(mse_list)}')
    model.tau = tau_best

    y_pred_test = model.predict(x_test)
    np.savetxt(pred_path, y_pred_test)

    mse = np.mean((y_pred_test - y_test) ** 2)
    print(f'Test set mse={mse}')


if __name__ == '__main__':
    main([0.03, 0.05, 0.1, 0.5, 1.0, 10.0], 'PS01\data\ds5_train.csv', 'PS01\data\ds5_valid.csv', 'PS01\data\ds5_test.csv',
    r'C:\Users\WIN10\Desktop\CS229\PS01\predict\problem5-(c).txt')