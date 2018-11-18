import numpy as np
from matplotlib import pyplot as plt

import math

X = np.array([400, 450, 900, 390, 550])

def calc_probs(x_vector, temp):
    alpha = np.min(x_vector)
    res = np.power(x_vector/alpha, -1 / temp)
    sigma = np.sum(res)
    return res / sigma


T = np.linspace(0.01,5,100)
P = np.zeros((len(T),len(X)))

for i,temp in enumerate(T):
    P[i,:] = calc_probs(X,temp)

print(P)

for i in range(len(X)):
    plt.plot(T, P[:, i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
