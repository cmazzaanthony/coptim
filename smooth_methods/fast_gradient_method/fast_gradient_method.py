import numpy as np


def funct(Q, c, x, gamma):
    return 0.5 * x.T.dot(Q).dot(x) + c + gamma


def dfunct(Q, c, x):
    return Q.dot(x) + c


def fast_gradient_method(x_0, delta, epsilon):
    Q = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, delta]])
    L = np.linalg.norm(Q)
    c = np.array([1, 1, 1, 1])
    x = x_0
    y = x_0
    alpha = 1
    k = 0
    while np.linalg.norm(dfunct(Q, c, x)) >= epsilon:
        prev_x = x
        x = y - 1 / L * dfunct(Q, c, y)

        prev_alpha = alpha
        alpha = (1 + np.sqrt(1 + 4 * np.power(prev_alpha, 2))) / 2

        y = x + (prev_alpha - 1) / alpha * (x - prev_x)

        k += 1

    print('Final parameters are \n'
          'x => {x}\n'
          'iterations => {k}\n'
          'delta => {delta}\n'
          'Q => \n {Q}'.format(x=x,
                               k=k,
                               delta=delta,
                               Q=Q))


if __name__ == '__main__':
    deltas = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    for delta in deltas:
        fast_gradient_method(x_0=np.array([1, 2, 3, 4]),
                        delta=delta,
                        epsilon=0.00001)
