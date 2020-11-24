import gzip
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
import numpy as np
from scipy.linalg import svd

from timeit import default_timer as timer

## Leemos los datos del fichero
f_pickle = "mnist_32.bnch"
f = gzip.open(f_pickle, mode='rb')
bnch = pickle.load(f)
f.close()
data = bnch['data']
N = data.shape[0]
M = data.shape[1]
data = np.array(data).reshape(N, M*M)

## Calculamos SVD de nuestra BBDD.
#U, s, VT = svd(data)
# print(s, s.shape)
NFEAT = 40

## Aplicamos PCA
start = timer()
pca = PCA(n_components=NFEAT)
reduced_data = pca.fit_transform(data)
# cov = pca.get_covariance()
recoverPCA = pca.inverse_transform(reduced_data)
recoverPCA = np.array(recoverPCA).reshape(N, M, M, 1)
# print(cov.shape)
# U, s, VT = svd(cov)
#print(s[:NFEAT*2]/745)
end = timer()
timePCA = (end - start) 
print(timePCA)

## Aplicamos NMF
start = timer()
model = NMF(n_components=NFEAT, init='random', random_state=0)
reduced_data = model.fit_transform(data)
recoverNMF = model.inverse_transform(reduced_data)
recoverNMF = np.array(recoverNMF).reshape(N, M, M, 1)
end = timer()
timeNMF = (end - start) 
print(timeNMF)

## Aplicamos ICA
start = timer()
transformer = FastICA(n_components=40, random_state=0)
reduced_data = transformer.fit_transform(data)
recoverICA = transformer.inverse_transform(reduced_data)
recoverICA = np.array(recoverICA).reshape(N, M, M, 1)
end = timer()
timeICA = (end - start) 
print(timeICA)