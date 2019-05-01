import numpy as np



class GradientMethodExactMinimization(Optimizer):

    def funct(Q, c, x, gamma):
        return 0.5 * x.T.dot(Q).dot(x) + c + gamma


    def dfunct(Q, c, x):
        return Q.dot(x) + c


    def exact_minimization(Q, c, x, d):
        g = dfunct(Q, c, x)
        return -1 * g.T.dot(d) / d.T.dot(Q).dot(d)


    def gradient_method(x_0, delta, epsilon):
        Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, delta]])
        c = np.array([1, 1, 1, 1])
        x = x_0
        k = 0
        while np.linalg.norm(dfunct(Q, c, x)) >= epsilon:
            # print(np.linalg.norm(dfunct(Q, c, x)))
            # print(k)
            descent_direction = -1 * dfunct(Q, c, x)

            step_size = exact_minimization(Q, c, x, descent_direction)

            # update step
            x = x + step_size * descent_direction

            k += 1

        print('Final parameters are \n'
              'x => {x}\n'
              'iterations => {k}\n'
              'delta => {delta}\n'
              'Q => \n {Q}'.format(x=x,
                                   k=k,
                                   delta=delta,
                                   Q=Q))


# if __name__ == '__main__':
#     deltas = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
#     for delta in deltas:
#         gradient_method(x_0=np.array([1, 2, 3, 4]),
#                         delta=delta,
#                         epsilon=0.00001)
