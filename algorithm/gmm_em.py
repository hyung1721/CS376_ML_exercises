import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mvn

np.random.seed(123)

# Data Generation
def GenerateData(mu1, mu2, sigma1, sigma2, g1, g2):
    data1 = np.random.multivariate_normal(mu1, sigma1, g1)
    data2 = np.random.multivariate_normal(mu2, sigma2, g2)
    return data1, data2

# Log Likelihood function
def LLH(D, M, C, G):
    n, p = D.shape
    k = len(G)
    output = 0.0

    for i in range(n):
        s = 0
        for j in range(k):
            s += G[j] * mvn(M[j], C[j]).pdf(D[i])
        output += np.log(s)
    
    return output

# Expectation Step
def E_step(D, M, C, G):
    n, p = D.shape
    k = len(G)
    E = np.zeros((k, n))

    for i in range(len(M)):
        for j in range(n):
            E[i, j] = G[i] * mvn(M[i], C[i]).pdf(D[j])
    
    E /= E.sum(0)

    return E

# Maximization Step
def M_step(D, E, G):
    n, p = D.shape
    k = len(G)

    G = np.zeros(k)
    for i in range(k):
        for j in range(n):
            G[i] += E[i, j] 
    G /= n

    M = np.zeros((k, p))
    for i in range(k):
        for j in range(n):
            M[i] += E[i, j] * D[j]
        M[i] /= E[i, :].sum()
    
    C = np.zeros((k, p, p))
    for i in range(k):
        for j in range(n):
            tmp = np.reshape(D[j] - M[i], (2, 1))
            C[i] += E[i, j] * np.dot(tmp, tmp.T)
        C[i] /= E[i, :].sum()
    
    return M, C, G

def plot(D, E, M, mu1, mu2, g1):
    cl1 = list()
    cl2 = list()

    true1 = 0
    true2 = 0

    for i in range(len(D)):
        if E[i, 0] > E[i, 1]:
            cl1.append(D[i, :])
            if i < g1:
                true1 += 1
            else:
                true2 += 1
        else:
            cl2.append(D[i, :])
            if i >= g1:
                true1 += 1
            else:
                true2 += 1
    
    cl1 = np.array(cl1)
    cl2 = np.array(cl2)
    index = 0 if true1 > true2 else 1
    color = ['y.', 'g.']

    print(f'EM has the accuracy of {float(max(true1, true2)) / len(D)}')
    print(f'Center: ({M[index,0]}, {M[index,1]}) and ({M[1-index,0]}, {M[1-index,1]})')

    plt.plot(cl1[:,0], cl1[:,1], color[index])
    plt.plot(cl2[:,0], cl2[:,1], color[1-index])
    plt.plot(M[0,0], M[0,1], marker='*', color='black', linestyle='None',label="Estimated center")
    plt.plot(M[1,0], M[1,1], marker='*', color='black', linestyle='None')
    plt.plot(mu1[0], mu1[1], marker='+', color='black', linestyle='None', label="Original center")
    plt.plot(mu2[0], mu2[1], marker='+', color='black', linestyle='None')
    plt.legend()
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('EM Algorithm using Gaussian Probability Density Functions')
    plt.show()

mu1 = [3, 0]
sigma1 = [[3, 0], [0, 0.5]]
mu2 = [-3, 0]
sigma2 = [[1, 0],[0, 2]]
g1 = 600
g2 = 400

data_1, data_2 = GenerateData(mu1, mu2, sigma1, sigma2, g1, g2)
D = np.concatenate((data_1, data_2), axis=0)

G = np.random.random(2)
G /= G.sum()
M = np.random.random((2, 2))
C = np.array([np.eye(2)] * 2)

Pnew = LLH(D, M, C, G)
Pold = 2 * Pnew

iteration = 0

while ((abs((Pold - Pnew) / Pnew) * 100) > 0.001 and (iteration <= 1000)):
    E = E_step(D, M, C, G)
    M, C, G = M_step(D, E, G)
    Pold = Pnew
    Pnew = LLH(D, M, C, G)
    iteration += 1

plot(D, E.T, M, mu1, mu2, g1)