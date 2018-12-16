import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def m_i(x, y, beta):
    return x @ beta - y


def update_beta(beta, p, step_size, m, x_i, lam):
    return beta - step_size * (x_i.T @ m + lam * beta)


def permute(X, y):
    idx = np.random.permutation([x for x in range(len(y))])
    return X[idx], y[idx]


def make_iterator(X, y, batch_size):
    for i in range(len(X)):
        u = min(i + batch_size, len(X))
        yield X[i:u], y[i:u]


def stochastic_gradient_method(beta_0, intercept, X, Y, step_size, num_epochs, lam, batch_size):
    beta = beta_0
    iterations = 0
    m = X.shape[0]
    p = X.shape[1] - 1
    error_per_epoch = 0

    losses = []
    for epoch in range(num_epochs):
        it = make_iterator(*permute(X, Y), batch_size)
        error_per_epoch = 0
        i = 0
        while True:
            try:
                x, y = next(it)
                f_value = m_i(x, y, beta)
                error_per_epoch += 0.5 * (f_value.T @ f_value + lam * beta.T @ beta)
                intercept = y.mean() - x.mean(0) @ beta  # intercept update
                beta = update_beta(beta, p, step_size, f_value, x, lam)
                i += len(y)
            except StopIteration as e:
                error_per_epoch /= i
                losses.append(error_per_epoch)
                # print('Epoch: %d, Total Error: %.3f' %(epoch, error_per_epoch))
                break
    return beta, np.stack(losses).reshape(len(losses))


if __name__ == '__main__':
    X = pd.read_csv('X.csv').values
    Y = pd.read_csv('Y.csv').values
    batch_sizes = [10, 20, 50, 100]
    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    beta_0 = np.random.normal(size=(X.shape[1], 1))
    intercept = 0
    f_star = 57.0410

    fig = plt.figure(figsize=(20, 8))
    plt.title(r"LASSO $f^k-f^*$")
    ax = fig.add_subplot(1, 1, 1)
    # fig.tight_layout()
    handles, labels = [], []
    ax.set_yscale('log')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Objective function error")
    for batch_size in batch_sizes:
        for lr in learning_rates:
            beta, losses = stochastic_gradient_method(beta_0=beta_0,
                                                      intercept=intercept,
                                                      X=X,
                                                      Y=Y,
                                                      step_size=lr,
                                                      batch_size=batch_size,
                                                      num_epochs=500,
                                                      lam=1)
            print('[[Batch size: %d learning rate: %f]]' % (batch_size, lr))
            p = ax.plot(losses - f_star)[0]
            handles.append(p)
            labels.append('Batch:%d Step:%.0e' % (batch_size, lr))
    plt.grid(True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
