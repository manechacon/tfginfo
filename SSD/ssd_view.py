import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import ssd_utils

import matplotlib

import gzip
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
import numpy as np
from scipy.linalg import svd
from tabulate import tabulate
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import imageio
from timeit import default_timer as timer
from sklearn.neural_network import MLPRegressor
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

img_shape = [69, 85]

# Recibe datos en formato fila y hace el reshape a imagen
def parse_img(data):
    n_time, _ = data.shape
    imgdata = data.reshape((n_time, img_shape[0], img_shape[1]))

    return imgdata

def pca_show_firstcomp(data, perc, n_show):
    cmap = plt.get_cmap('YlOrRd')
    pca = PCA(n_components =perc, svd_solver = 'full')
    reduced_data = pca.fit_transform(data)
    recover = pca.inverse_transform(reduced_data)
    n_comp, _ = pca.components_.shape
    components = parse_img(pca.components_)
    for i in range (n_show):
        plt.imshow(components[i, :, :], origin = "lower", cmap = cmap)
        plt.title('Componente '+str(i+1) + ' de ' + str(n_comp))
        plt.savefig("components/%.2f" % (perc) +'_'+ str(i+1) +".png" )
    pass
    
def create_animation_pca(df, perc, start_hour, end_hour, fps = 2):
    cmap = plt.get_cmap('YlOrRd')
    
    data = ssd_utils.extract_data(df)
    
    pca = PCA(n_components = perc, svd_solver = 'full')
    reduced_data = pca.fit_transform(data)
    recover = pca.inverse_transform(reduced_data)
    error = mean_squared_error(data, recover)
    maerror = mean_absolute_error(data, recover)
    r2 = r2_score(data, recover, multioutput='variance_weighted')
    n_comp, _ = pca.components_.shape
    
    imgdata = parse_img(data)
    imgdatarecover = parse_img(recover)
    
    for i in range(start_hour, end_hour):
        plt.imshow(imgdata[i, :, :], origin = "lower", cmap = cmap)
        plt.title(df.index[i])
        plt.savefig("originalpng/%d" % (i+1) +".png" )

        plt.imshow(imgdatarecover[i, :, :], origin = "lower", cmap = cmap)
        plt.title(df.index[i])
        plt.savefig("recoverpng/%d" % (i+1) +".png" )

    imglist = []
    recoverlist = []
    for i in range(end_hour - start_hour):
        imglist.append("originalpng/%d" % (i+1)+".png")
        recoverlist.append("recoverpng/%d" % (i+1)+".png")

    with imageio.get_writer('gifts/pca_original_'+ str(df.index[start_hour]) +'_' + str(df.index[end_hour-1]) +'.gif', mode='I', fps = fps) as writer:
        for filename in imglist:
            image = imageio.imread(filename)
            writer.append_data(image)

    with imageio.get_writer('gifts/pca_recover_'+ str(df.index[start_hour]) +'_' + str(df.index[end_hour-1])+ '.gif', mode='I', fps = fps) as writer:
        for filename in recoverlist:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    pass
