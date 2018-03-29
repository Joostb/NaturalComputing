import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

def formula(err):
    return np.log((1-err)/err)/2


def main():
    errors = np.arange(1e-4,1,0.01)
    betas = formula(errors)



    plt.plot(errors,betas)
    plt.xlabel("Error")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
