import numpy as np
from matplotlib import pyplot as plt

import math

X = np.array([400, 450, 900, 390, 550])

def calc_probs(x_vector, temp):
    res = np.zeros(len(x_vector))
    alpha = min(x_vector)
    for i,x in enumerate(x_vector):
        prob = math.pow((x/alpha), -1/temp)
        res[i] = prob

    sigma = sum(res)
    for i in range(len(x_vector)):
        res[i] = res[i] / sigma

    return res


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
