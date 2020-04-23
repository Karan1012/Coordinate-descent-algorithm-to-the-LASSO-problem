# referred from https://xavierbourretsicotte.github.io/lasso_implementation.html
import numpy as np
from numpy import genfromtxt
import time

time_start = time.perf_counter()

data = genfromtxt(r'Xmat.csv', delimiter=',')
data_y = genfromtxt(r'Ymat.csv', delimiter=',')

x_train = data[1:202, 1:10001]
y_train = data_y[1:202, 1:2]


def soft_threshold(rho, gamma):
    if rho < -gamma:
        return rho + gamma
    elif rho > gamma:
        return rho - gamma
    else:
        return 0


m, n = x_train.shape
x_train = x_train / (np.linalg.norm(x_train, axis=0))


def coordinate_descent(beta, x_train, y_train, gamma, iters):
    # Looping on the number of iterations
    for i in range(iters):
        # Looping on each coordinate
        for j in range(n):
            X_j = x_train[:, j].reshape(-1, 1)
            ypred = x_train @ beta
            rho = X_j.T @ (y_train - ypred + beta[j] * X_j)

            beta[j] = soft_threshold(rho, gamma)

    return beta.flatten()


beta_0 = np.zeros((n, 1))
beta = np.zeros((n, 1))

beta = coordinate_descent(beta_0, x_train, y_train, gamma=16, iters=2)
non_zero = np.count_nonzero(beta)
print(beta[np.argsort(beta)[-20:]][::-1])

print('Non zero beta values = ', non_zero)
time_elapsed = (time.perf_counter() - time_start)
print('time elapsed = ', time_elapsed)
