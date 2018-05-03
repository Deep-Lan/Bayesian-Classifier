import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
    g = np.exp(- (((x-mu) / sigma) ** 2) / 2) / (sigma * np.sqrt(2 * np.pi))
    return g


def main():
    # dataset
    obser_dataset1 = np.array([-3.9847, -3.5549, -1.2401, -0.9780, 0.7932, -2.8531, -2.7605, -3.7287,
                              -3.5414, -2.2692, -3.4549, -3.0752, -3.9934, -0.9780, -1.5799, -1.4885,
                              -0.7431, -0.4221, -1.1186, -2.3462, -1.0826, -3.4196, -1.3193, -0.8367, -0.6579, -2.9683])
    obser_dataset2 = np.array([2.8792, 0.7932, 1.1882, 3.0682, 4.2532, 0.3271, 0.9846, 2.7648, 2.6588])
    x = np.linspace(-5, 5, 100)

    # likelihold
    mean1 = np.mean(obser_dataset1)
    std1 = np.std(obser_dataset1)
    mean2 = np.mean(obser_dataset2)
    std2 = np.std(obser_dataset2)
    p_x_omega1 = gaussian(x, mean1, std1)
    p_x_omega2 = gaussian(x, mean2, std2)

    # posterior
    p_omega1 = 0.9
    p_omega2 = 0.1
    p_omega1_x = p_omega1 * p_x_omega1 / (p_omega1 * p_x_omega1 + p_omega2 * p_x_omega2)
    p_omega2_x = p_omega2 * p_x_omega2 / (p_omega1 * p_x_omega1 + p_omega2 * p_x_omega2)

    # loss
    lambda12 = 1
    lambda21 = 6
    R1 = lambda12 * p_omega2_x
    R2 = lambda21 * p_omega1_x

    # decide result
    decision_result = []
    for i in range(len(x)):
        if R1[i] < R2[i]:
            decision_result.append(1)
        else:
            decision_result.append(2)
    print(decision_result)

    # show image
    plt.figure(1)
    plt.title('likelihood')
    plt.plot(x, p_x_omega1, color='red', label='p_x_omega1')
    plt.plot(x, p_x_omega2, color='blue', label='p_x_omega2')
    plt.legend(loc='upper right')
    plt.figure(2)
    plt.title('posterior')
    plt.plot(x, p_omega1_x, color='red', label='p_omega1_x')
    plt.plot(x, p_omega2_x, color='blue', label='p_omega2_x')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()