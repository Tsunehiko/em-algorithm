import numpy as np
from scipy.stats import norm

tol = 1e-12

open_data = np.array([8.0, 3.0, 6.0, 9.0, 7.0, 6.0, 2.0, 6.0, 4.0, 4.0, 5.0, 7.0])
component = 3 #ガウス分布の数
pi = np.ones(component) / component #混合係数
mu = np.array([8.0,3.0,6.0]) #平均
sigma = np.array([1.0,1.0,1.0]) #分散
N = open_data.size
K = pi.size
gamma = np.zeros((K, N)) #事後確立

log_likelihood = - np.float("inf")

while True:

    '''Eステップ'''
    for n in range(N):
        likelihood = np.array([pi[j] * norm.pdf(x=open_data[n],loc=mu[j],scale=sigma[j]) for j in range(K)])
        prior_prob = np.sum(likelihood)
        gamma.T[n] = likelihood / prior_prob

    '''Mステップ'''
    for k in range(K):
        gamma_sum = np.sum(gamma[k])
        mu[k] = np.sum([gamma[k][n] * open_data[n] for n in range(N)]) / gamma_sum
        sigma[k] = np.sum([gamma[k][n] * (open_data[n] - mu[k]) ** 2 for n in range(N)]) /gamma_sum
        pi[k] = gamma_sum / N

    log_likelihood_new = 0
    for n in range(N):
        log_likelihood_new += np.log(np.sum([norm.pdf(x=open_data[n],loc=mu[k],scale=sigma[k]) for k in range(K)]))

    if log_likelihood_new - log_likelihood < tol:
        break

    log_likelihood = log_likelihood_new

for k in range(K):
    print(k)
    print("pi:{},mi:{},sigma:{}".format(pi[k],mu[k],sigma[k]))