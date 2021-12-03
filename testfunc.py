import numpy as np

from utils import transform_data, get_confusion_matrix
from EMfunctions import spectralEM


df = np.loadtxt('data/bluebird_crowd.txt')
df = transform_data(df)

init_mu = get_confusion_matrix(k=2,labels=df)
print(init_mu)

EM_optimizor = spectralEM(init_mu=init_mu, labels=df)
EM_optimizor.run()
print(EM_optimizor.output_mu())

