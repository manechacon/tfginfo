import gzip
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.linalg import svd
from tabulate import tabulate

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from timeit import default_timer as timer

import logging
from time import time

from numpy.random import RandomState

from sklearn.datasets import fetch_olivetti_faces

from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition
import matplotlib as mpl

def autoval_analisis(data, n_comp):
    data = data - np.mean(data)
    cov = np.matmul(data.T, data) # Matriz de covarianzas

    _, s, _ = svd(cov)
    
    plt.yscale("log")
    plt.xscale("log")
    plt.plot((s[:-5]))
    plt.axvline(x=n_comp, ymin=10.**-2, ymax=10.**1, color = "red")
    
    Total = sum(s)

    print("Tomamos %d componentes" % (n_comp) + " y nos queda un %2.2f" % (100*(sum(s[:n_comp])/Total)) +  " % de la varianza total")
    

def oliv_autoval_analisis(n_comp = 40):
    rng = RandomState(0)
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True,
                                    random_state=rng)
    data = faces
    
    autoval_analisis(data, n_comp)
    
    pass

def mnist_autoval_analisis(n_comp = 50):
    f_pickle = "mnist_32.bnch"
    f = gzip.open(f_pickle, mode='rb')
    bnch = pickle.load(f)
    f.close()
    data = bnch['data_test']
    N = data.shape[0]
    M = data.shape[1]
    data = np.array(data).reshape(N, M*M)
    
    autoval_analisis(data, n_comp)
    
    pass

    
def oliv_pretest(n_comp = 60):
    rng = RandomState(0)

    n_row, n_col = 1,5
    n_components = n_comp
    image_shape = (64, 64)
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True,
                                    random_state=rng)
    data = faces
    scaler = StandardScaler()



    n_samples, n_features = data.shape
                   
    def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=mpl.cm.magma):
        plt.figure(figsize=(2. * n_col, 2.36 * n_row))
       
        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=cmap,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)             

    # #############################################################################
    # List of the different estimators, whether to center and transpose the
    # problem, and whether the transformer uses the clustering API.
    estimators = [
        ('PCA using randomized SVD',
         decomposition.PCA(n_components=n_components, svd_solver='randomized',
                           whiten=True),
         False),

        ('Non-negative components - NMF',
         decomposition.NMF(n_components=n_components,  init='random', random_state=0,  tol=5e-3),
         False),

        ('Independent components - FastICA',
         decomposition.FastICA(n_components=n_components, whiten=True),
         False),
    ]

    # #############################################################################
    # Plot a sample of the input data
    nplot = 5

    # #############################################################################
    # Do the estimation and plot it

    for name, estimator, center in estimators:
        print("Extracting the top %d %s..." % (n_components, name))
        t0 = time()
        if center:
            data = faces
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        else:
            data = faces
        estimator.fit(data)
        train_time = (time() - t0)
        data_transform = estimator.transform(data)
        recover = estimator.inverse_transform(data_transform)
        error = mean_squared_error(data, recover)
        print("Error cuadrático medio en reconstrucción: ", error)
        print("done in %0.3fs" % train_time)
        if hasattr(estimator, 'cluster_centers_'):
            components_ = estimator.cluster_centers_
        else:
            components_ = estimator.components_

        # Plot an image representing the pixelwise variance provided by the
        # estimator e.g its noise_variance_ attribute. The Eigenfaces estimator,
        # via the PCA decomposition, also provides a scalar noise_variance_
        # (the mean of pixelwise variance) that cannot be displayed as an image
        # so we skip it.
        if (hasattr(estimator, 'noise_variance_') and
                estimator.noise_variance_.ndim > 0):  # Skip the Eigenfaces case
            plot_gallery("Pixelwise variance",
                         estimator.noise_variance_.reshape(1, -1), n_col=1,
                         n_row=1)

        plot_gallery("First 5 faces", data[:nplot])

        plot_gallery('%s - Recover - Train time %.1fs - MSE %.5f' % (name, train_time, error),
                     recover[:nplot])

        plot_gallery('%s - First 5 components' % (name),
                     components_[:nplot])


        
        fig, ax = plt.subplots(figsize=(15, 2))
        fig.subplots_adjust(bottom=0.5)

        cmap = mpl.cm.magma
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=ax, orientation='horizontal', label='Valor de Bit')

        plt.show()
        print("\n\n")
    pass

def minist_pretest(n_comp = 60):
    rng = RandomState(0)
    nplot = 5
    n_row, n_col = 1,5
    image_shape = (32, 32)

    f_pickle = "mnist_32.bnch"
    f = gzip.open(f_pickle, mode='rb')
    bnch = pickle.load(f)
    f.close()
    data = bnch['data_test']
    # target = bnch['target_test']
    N = data.shape[0]
    M = data.shape[1]
    data = np.array(data).reshape(N, M*M)



    def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=mpl.cm.Greys):
        plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=cmap,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

    # #############################################################################
    # List of the different estimators, whether to center and transpose the
    # problem, and whether the transformer uses the clustering API.
    estimators = [
        ('PCA using randomized SVD',
         decomposition.PCA(n_components=n_comp, svd_solver='randomized',
                           whiten=True),
         False),

        ('Non-negative components - NMF',
         decomposition.NMF(n_components=n_comp,  init='random', random_state=0,  tol=5e-3),
         False),

        ('Independent components - FastICA',
         decomposition.FastICA(n_components=n_comp, whiten=True),
         False),
    ]


    for name, estimator, center in estimators:
        print("Extracting the top %d %s..." % (n_comp, name))
        t0 = time()

        if center:
            scaler = StandarScaler()
            data = scaler.fit_transform(data)

        estimator.fit(data)
        train_time = (time() - t0)
        data_transform = estimator.transform(data)
        recover = estimator.inverse_transform(data_transform)
        error = mean_squared_error(data, recover)
        # errorn = number_error(recover, data)
        # print(errorn)
        # print(np.mean(errorn))
        print("Error cuadrático medio en reconstrucción: ", error)
        print("done in %0.3fs" % train_time)
        if hasattr(estimator, 'cluster_centers_'):
            components_ = estimator.cluster_centers_
        else:
            components_ = estimator.components_

        # Plot an image representing the pixelwise variance provided by the
        # estimator e.g its noise_variance_ attribute. The Eigenfaces estimator,
        # via the PCA decomposition, also provides a scalar noise_variance_
        # (the mean of pixelwise variance) that cannot be displayed as an image
        # so we skip it.
        if (hasattr(estimator, 'noise_variance_') and
                estimator.noise_variance_.ndim > 0):  # Skip the Eigenfaces case
            plot_gallery("Pixelwise variance",
                         estimator.noise_variance_.reshape(1, -1), n_col=1,
                         n_row=1)
        if name == "PCA using randomized SVD":
            plot_gallery("First 5 numbers", data[:nplot])

            plot_gallery('%s - Recover - Train time %.1fs - MSE %.2f' % (name, train_time, error),
                         recover[:nplot])

            plot_gallery('%s - First 6 components' % (name),
                         components_[:nplot])
        print("\n\n")
        

    # fig, ax = plt.subplots(figsize=(15, 2))
    # fig.subplots_adjust(bottom=0.5)

    # cmap = mpl.cm.cool
    # norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #              cax=ax, orientation='horizontal', label='Valor de Bit')

    plt.show()
    

    
def oliv_aetest(n_comp = 40):
    nplot = 5
    rng = RandomState(0)
    n_row, n_col = 1,5
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True,
                                    random_state=rng)
    image_shape = (64, 64)
    
    def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=mpl.cm.magma):
        plt.figure(figsize=(2. * n_col, 2.36 * n_row))

        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=cmap,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        
        pass

    data = faces
    
    mlp = MLPRegressor(hidden_layer_sizes = (n_comp), 
                            solver = 'adam',
                            activation = 'identity', 
                            random_state=1, 
                            tol = 1.e-6, 
                            n_iter_no_change = 10,
                            max_iter=1000, 
                            shuffle = True,
                            early_stopping = False,
                            verbose = False)

    regr = Pipeline(steps=[('std_sc', StandardScaler()),
                           ('mlp', mlp)])

    y_transformer = StandardScaler()
    linear_ae_clf =  TransformedTargetRegressor(regressor=regr,
                                                 transformer=y_transformer)

    mlp = MLPRegressor(hidden_layer_sizes = (100, n_comp, 100), 
                                solver = 'adam',
                                activation = 'relu', 
                                random_state=1, 
                                tol = 1.e-6, 
                                n_iter_no_change = 10,
                                max_iter=1000, 
                                shuffle = True,
                                early_stopping = False,
                                verbose = False)

    regr = Pipeline(steps=[('std_sc', StandardScaler()),
                           ('mlp', mlp)])

    y_transformer = StandardScaler()
    multilayer_ae_clf =  TransformedTargetRegressor(regressor=regr,
                                                 transformer=y_transformer)
    
    linear_ae_clf.fit(data, data)
    recover_lin = linear_ae_clf.predict(data)
    print("Error cuadrático medio de arquitectura lineal: ", mean_squared_error(recover_lin, data))
    
    multilayer_ae_clf.fit(data, data)
    recover_multi = multilayer_ae_clf.predict(data)
    print("Error cuadrático medio de arquitectura multicapa: ", mean_squared_error(recover_multi, data))
    
    plot_gallery(f"First {nplot} faces", faces[:nplot])
    plot_gallery('Recover-lineal-ae', recover_lin[:nplot])
    plot_gallery('Recover-multilayer-ae', recover_multi[:nplot])
    
    pass

def mnist_aetest(n_comp = 60):
    nplot = 5
    rng = RandomState(0)
    n_row, n_col = 1,5
    image_shape = (32, 32)
    f_pickle = "mnist_32.bnch"
    f = gzip.open(f_pickle, mode='rb')
    bnch = pickle.load(f)
    f.close()
    data = bnch['data_test']
    N = data.shape[0]
    M = data.shape[1]
    data = np.array(data).reshape(N, M*M)
    
    
    def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=mpl.cm.cool):
        plt.figure(figsize=(2. * n_col, 2.36 * n_row))

        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=cmap,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        
        pass

    
    mlp_lin = MLPRegressor(hidden_layer_sizes = (n_comp), 
                            solver = 'adam',
                            activation = 'identity', 
                            random_state=1, 
                            tol = 1.e-6, 
                            alpha = 10.**-9,
                            n_iter_no_change = 500,
                            max_iter = 5000, 
                            shuffle = True,
                            early_stopping = False,
                            verbose = False)

    mlp_multi = MLPRegressor(hidden_layer_sizes = (300, n_comp, 300), 
                                solver = 'adam',
                                activation = 'relu', 
                                random_state=1, 
                                tol = 1.e-6, 
                                alpha = 10.**-9,
                                n_iter_no_change = 500,
                                max_iter = 5000, 
                                shuffle = True,
                                early_stopping = False,
                                verbose = False)

    regr_lin = Pipeline(steps=[('std_sc', StandardScaler()),
                           ('mlp_lin', mlp_lin)])

    regr_multi = Pipeline(steps=[('std_sc', StandardScaler()),
                           ('mlp_multi', mlp_multi)])

    y_transformer = StandardScaler()

    lin_ae_clf =  TransformedTargetRegressor(regressor=regr_lin,
                                                 transformer=y_transformer)

    multi_ae_clf =  TransformedTargetRegressor(regressor=regr_multi,
                                             transformer=y_transformer)

    y_transformer = StandardScaler()
    
    lin_ae_clf.fit(data, data)
    # recover_lin = lin_ae_clf.predict(data)
    
    file_lin = open(f"stored_objs/store_lin_{n_comp}.pickle", "wb")
    pickle.dump(lin_ae_clf, file_lin)
    file_lin.close()
    # print("Error cuadrático medio de arquitectura lineal: ", mean_squared_error(recover_lin, data))
    
    
    multi_ae_clf.fit(data, data)
    # recover_multi = multi_ae_clf.predict(data)
    
    file_mul = open(f"stored_objs/store_mul_{n_comp}.pickle", "wb")
    pickle.dump(multi_ae_clf, file_mul)
    file_mul.close()
    # print("Error cuadrático medio de arquitectura multicapa: ", mean_squared_error(recover_multi, data))
    
    
    """
    plot_gallery(f"First {nplot} faces", data[:nplot])
    plot_gallery('Recover-lineal-ae', recover_lin[:nplot])
    plot_gallery('Recover-multilayer-ae', recover_multi[:nplot])"""
    
    pass
        
       