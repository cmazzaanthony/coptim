import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

np.set_printoptions(suppress=True)

def mse(x, beta, y,intercept=None):
    if intercept is not None:
        x = np.concatenate([np.ones((x.shape[0],1)),x],axis=1)
        beta = np.concatenate([[intercept],beta],axis=0)
    diff = x @ beta - y
    return (1/len(y) * 0.5 * diff.T @ diff)[0][0]


def group_lasso_regularizer(beta, weights):
    beta = group_betas(beta)
    return sum([weights[j] * np.linalg.norm(beta[j]) for j in range(len(beta))])


def mse_grad(x, beta, y):
    return 1/len(y)*-x.T @ (y-x @ beta)


def prox_operator_glasso(beta, lam, weights,step_size,type='group'):
    beta = group_betas(beta) if type == 'group' else beta
    new_betas = []
    for j in range(len(beta)):
        new_beta_j = beta[j] * np.maximum(1 - step_size *lam * weights[j] / np.linalg.norm(beta[j]),0)
        new_betas.append(new_beta_j)
    new_betas = ungroup_betas(new_betas)
    return new_betas



def initialize_group_lasso_weights(beta_0):
    return np.array(list(map(lambda x:np.sqrt(x.size),beta_0)))

def group_betas(beta_0):
    beta_0 = [
        beta_0[0].reshape(1, 1),  # age (1)
        beta_0[1].reshape(1, 1),  # sex (1)
        beta_0[2:7],  # Jitters (5)
        beta_0[7:13],  # Shimmer (6)
        beta_0[13:15],  # NHR, HNR (2)
        beta_0[15].reshape(1, 1),  # RPDE (1)
        beta_0[16].reshape(1, 1),  # DFA (1)
        beta_0[17].reshape(1, 1)  # PPE (1)
    ]

    return beta_0


def ungroup_betas(beta):
    single_beta = list()
    for j in range(len(beta)):
        single_beta += list(beta[j].flatten())

    return np.array(single_beta).reshape(len(single_beta), 1)

def print_group(betas):
    print(r'Group $j$ & $\hat{\beta}_{(j)}')
    for i,beta in enumerate(group_betas(betas)):
        try:
            beta = list(map(lambda x:str(np.round(x,3)),beta.flatten()))
        except AttributeError:
            beta = list(map(lambda x:str(np.round(x,3)),beta))
        print("%d & %s \\\\"%(i+1,' '.join(beta)))

def permute(X,y):
    idx = np.random.permutation([x for x in range(len(y))])
    return X[idx],y[idx]

def train_test_split(X,y,r):
    X,y = permute(X,y)
    l = int(r * len(y))
    return X[:l],y[:l],X[l:],y[l:]

def proximal_grad_descent(beta_0, weights, X, Y, step_size, max_iter, lam,type):
    beta = beta_0
    iterations = 0
    include_int = False

    X_train, y_train, X_test, y_test = train_test_split(X,Y,r=0.8)

    fs_train,fs_test = [],[]
    Y_bar = Y.mean()
    X_bar = X.mean(0)
    Y_tilde = y_train if not include_int else y_train
    X_tilde = X_train if not include_int else  (X_train-X_bar)#/X.std(0)
    for iterations in range(max_iter):
        sgd_step = beta - step_size * mse_grad(X_tilde, beta, Y_tilde)
        beta = prox_operator_glasso(sgd_step,lam,weights,step_size,type)
        intercept = None if not include_int else Y_bar - X_bar @ beta
        f_train = mse(X_tilde, beta, Y_tilde,intercept) + group_lasso_regularizer(beta, weights)
        f_test = mse(X_test, beta, y_test,intercept) + group_lasso_regularizer(beta, weights)
        fs_train.append(f_train)
        fs_test.append(f_test)
        print("Obj: {}".format(f_train))
    print("Train f: {}".format(f_train))
    print("Test f: {}".format(f_test))
    return intercept,beta,np.array(fs_train),np.array(fs_test)

if __name__ == '__main__':
    X = pd.read_csv('X.csv').values
    Y = pd.read_csv('Y.csv').values
    beta_0 = np.random.randn(X.shape[1], 1)
    m = X.shape[0]
    p = X.shape[1]
    f_star = 49.9649
    # beta_0 = group_betas(beta_0)
    group_weights = initialize_group_lasso_weights(beta_0)
    J = len(beta_0)

    # Group LASSO

    intercept,beta,f_train,f_test = proximal_grad_descent(beta_0=beta_0,
                          weights=group_weights,
                          X=X,
                          Y=Y,
                          step_size=0.005,
                          max_iter=10000,
                          lam=0.02,
                          type="group")
    
    # fig = plt.figure()
    # plt.title(r"Group LASSO $f^k-f^*$")
    # ax = fig.add_subplot(1, 1, 1)
    # p1 = ax.plot(f_train-f_star)
    # p2 = ax.plot(f_test-f_star)
    # handles = [p1[0],p2[0]]
    # labels = ['Train','Test']
    # ax.set_yscale('log')
    # ax.set_xlabel("Steps")
    # ax.set_ylabel("Objective function error")
    # print(intercept)
    # print(beta)
    # print_group(beta)
    # plt.grid(True)
    # ax.legend(handles, labels)
    # plt.show()

    # LASSO 

    intercept,beta,f_train,f_test = proximal_grad_descent(beta_0=beta_0,
                          weights=[1 for _ in range(J)],
                          X=X,
                          Y=Y,
                          step_size=0.005,
                          max_iter=10000,
                          lam=0.02,
                          type="single")
    
    fig = plt.figure()
    plt.title(r"LASSO $f^k-f^*$")
    ax = fig.add_subplot(1, 1, 1)
    p1 = ax.plot(f_train-f_star)
    p2 = ax.plot(f_test-f_star)
    handles = [p1[0],p2[0]]
    labels = ['Train','Test']
    ax.set_yscale('log')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Objective function error")
    print(intercept)
    print(beta)
    print_group(beta)
    plt.grid(True)
    ax.legend(handles, labels)
    plt.show()