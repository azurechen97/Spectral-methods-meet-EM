import numpy as np

from utils import transform_data, get_confusion_matrix, errorRate
from EMfunctions import spectralEM


df = np.loadtxt('data/rte_crowd.txt')
df = transform_data(df)

truth = np.loadtxt('data/rte_truth.txt')

init_mu = get_confusion_matrix(k=2, labels=df)
# print(init_mu)

EM_optimizor = spectralEM(init_mu=init_mu, labels=df)
EM_optimizor.run(strategy='max_iter', max_iter=10)
# print(EM_optimizor.output_mu())
# print(EM_optimizor.output_q())

error = errorRate(EM_optimizor.output_q(), truth)
print(error)