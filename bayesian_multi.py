import numpy as np


def gaussian(X, Mu, Sigma):
    '''
    :param X: of shape (n, 1)
    :param Mu: of shape (n, 1)
    :param Sigma: of shape (n, n)
    :return: probability in X
    '''
    n, _ = X.shape
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_det = np.linalg.det(Sigma)
    g = np.exp(np.dot(np.dot((X-Mu).T, Sigma_inv), X-Mu)/(-2)) / (np.sqrt(Sigma_det) * (np.sqrt(2 * np.pi) ** n))
    return g


def main():
    # dataset
    x_1dim = np.random.normal(3, 2, 100).reshape((1, 100))
    x_2dim = np.random.normal(2, 2, 100).reshape((1, 100))
    X_class1 = np.vstack((x_1dim, x_2dim)).T
    x_1dim = np.random.normal(-2, 2, 100).reshape((1, 100))
    x_2dim = np.random.normal(2, 1, 100).reshape((1, 100))
    X_class2 = np.vstack((x_1dim, x_2dim)).T
    x_1dim = np.random.normal(0, 1, 100).reshape((1, 100))
    x_2dim = np.random.normal(-1, 1, 100).reshape((1, 100))
    X_class3 = np.vstack((x_1dim, x_2dim)).T
    x_predict = np.array([0, -1]).reshape((2, 1))   # predicted sample point,you could select [3, 2] [-2, 2] [0, -1] or others

    # likelihold
    mean1 = np.mean(X_class1, axis=0).reshape(2, 1)
    cov1 = np.cov(X_class1, rowvar=False)
    mean2 = np.mean(X_class2, axis=0).reshape(2, 1)
    cov2 = np.cov(X_class2, rowvar=False)
    mean3 = np.mean(X_class3, axis=0).reshape(2, 1)
    cov3 = np.cov(X_class3, rowvar=False)
    p_x_omega1 = gaussian(x_predict, mean1, cov1)
    p_x_omega2 = gaussian(x_predict, mean2, cov2)
    p_x_omega3 = gaussian(x_predict, mean3, cov3)

    # posterior
    p_omega1 = 0.3
    p_omega2 = 0.2
    p_omega3 = 0.5
    p_omega1_x = p_omega1 * p_x_omega1 / (p_omega1 * p_x_omega1 + p_omega2 * p_x_omega2 + p_omega3 * p_x_omega3)
    p_omega2_x = p_omega2 * p_x_omega2 / (p_omega1 * p_x_omega1 + p_omega2 * p_x_omega2 + p_omega3 * p_x_omega3)
    p_omega3_x = p_omega3 * p_x_omega3 / (p_omega1 * p_x_omega1 + p_omega2 * p_x_omega2 + p_omega3 * p_x_omega3)
    p_omega_x = np.array([p_omega1_x, p_omega2_x, p_omega3_x]).reshape((3, 1))

    # loss
    loss_para = np.array([0, 1, 2, 1, 0, 3, 2, 2, 0]).reshape((3, 3))
    R = np.dot(loss_para, p_omega_x)

    # decide result
    select_result = np.where(R == np.min(R))[0][0] + 1
    print(select_result)


if __name__ == '__main__':
    main()