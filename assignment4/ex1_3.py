import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def bootstrap(observations, N):
    """
    Taking N bootstrap samples from observations
    :param observations:
    :param N: sample size
    :return: sample
    """
    return np.random.choice(observations, N, True)


def formula(N):
    return (1 - (1/N))**N


def bagging_left_out(N, n_iters=1000):
    dataset = np.arange(0, N)
    left_out = []

    for _ in range(n_iters):
        sample = bootstrap(dataset, N)
        left_out.append(np.mean([0 if number in sample else 1 for number in dataset]))

    mean_left_out = [np.mean(left_out[:i]) for i in range(len(left_out))]

    # plt.figure()
    # plt.plot([formula(N) for _ in range(len(left_out))], label="Actual")
    # plt.plot(mean_left_out, label="Approximation")
    # plt.legend()
    # plt.show()

    return mean_left_out


def main():
    Ns = [2, 5, 10, 50, 100, 1000]

    plt.figure()
    for N in tqdm(Ns):
        plt.plot(bagging_left_out(N), label="N = {}; exact = {:.2f}".format(N, formula(N)))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
