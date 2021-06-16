import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

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
import matplotlib as mpl

img_shape = [69, 85]

# Lee los datos del año 2020 (cbt), devuelve dataframe.
def read_data_2020():
    df1 = pd.read_csv('data/ssd_pen_large_9_jan_2020.nc.csv', sep = ";")
    df2 = pd.read_csv('data/ssd_pen_large_9_feb_2020.nc.csv', sep = ";")
    df3 = pd.read_csv('data/ssd_pen_large_9_mar_2020.nc.csv', sep = ";")
    df4 = pd.read_csv('data/ssd_pen_large_9_apr_2020.nc.csv', sep = ";")
    df5 = pd.read_csv('data/ssd_pen_large_9_may_2020.nc.csv', sep = ";")
    df6 = pd.read_csv('data/ssd_pen_large_9_jun_2020.nc.csv', sep = ";")
    df7 = pd.read_csv('data/ssd_pen_large_9_jul_2020.nc.csv', sep = ";")
    df8 = pd.read_csv('data/ssd_pen_large_9_aug_2020.nc.csv', sep = ";")
    df9 = pd.read_csv('data/ssd_pen_large_9_sep_2020.nc.csv', sep = ";")
    df10 = pd.read_csv('data/ssd_pen_large_9_oct_2020.nc.csv', sep = ";")
    df11 = pd.read_csv('data/ssd_pen_large_9_nov_2020.nc.csv', sep = ";")
    df12 = pd.read_csv('data/ssd_pen_large_9_dec_2020.nc.csv', sep = ";")

    frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]
    df = pd.concat(frames, ignore_index=True)
    
    return df

def read_data_2019():
    df1 = pd.read_csv('data/ssd_pen_large_9_jan_2019.nc.csv', sep = ";")
    df2 = pd.read_csv('data/ssd_pen_large_9_feb_2019.nc.csv', sep = ";")
    df3 = pd.read_csv('data/ssd_pen_large_9_mar_2019.nc.csv', sep = ";")
    df4 = pd.read_csv('data/ssd_pen_large_9_apr_2019.nc.csv', sep = ";")
    df5 = pd.read_csv('data/ssd_pen_large_9_may_2019.nc.csv', sep = ";")
    df6 = pd.read_csv('data/ssd_pen_large_9_jun_2019.nc.csv', sep = ";")
    df7 = pd.read_csv('data/ssd_pen_large_9_jul_2019.nc.csv', sep = ";")
    df8 = pd.read_csv('data/ssd_pen_large_9_aug_2019.nc.csv', sep = ";")
    df9 = pd.read_csv('data/ssd_pen_large_9_sep_2019.nc.csv', sep = ";")
    df10 = pd.read_csv('data/ssd_pen_large_9_oct_2019.nc.csv', sep = ";")
    df11 = pd.read_csv('data/ssd_pen_large_9_nov_2019.nc.csv', sep = ";")
    df12 = pd.read_csv('data/ssd_pen_large_9_dec_2019.nc.csv', sep = ";")

    frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]
    df = pd.concat(frames, ignore_index=True)
    
    return df

# Recibe dataframe y lo interpola a datos horarios, devuelve dataframe.
def interpolate_1hour(df):
    df = df.set_index(pd.to_datetime(df['time']))
    df_resampled = df.resample('60min').asfreq()
    df_resampled = df_resampled.interpolate()
    
    return df_resampled


# Recibe dataframe y devuelve los datos en formato de matriz (numpy array).
def extract_data(df):
    data = df[df.columns[1:]].to_numpy()
    
    return data


def scale_data(data):
    scaler = StandardScaler()
    data_scale = scaler.fit_transform(data)
    
    return data_scale

def get_all_data_2020(norm = False, itp = True):
    df = read_data_2020()
    
    if itp:
        df = interpolate_1hour(df)
    
    if norm:
        data = scale_data(extract_data(df))
    else:
        data = extract_data(df)
    
    return data

def get_all_data_2019(norm = False, itp = True):
    df = read_data_2019()
    
    if itp:
        df = interpolate_1hour(df)
    
    if norm:
        data = scale_data(extract_data(df))
    else:
        data = extract_data(df)
    
    return data


def train_test_data_2020(norm = False):
    df_orig = read_data_2020()
    df_interp = interpolate_1hour(df_orig)
    df_train = df_interp[(df_interp.index < '2020-08-1 00:00:00')]
    df_test = df_interp[(df_interp.index >= '2020-08-1 00:00:00')]
    
    if norm:
        data_train = scale_data(extract_data(df_train))
        data_test = scale_data(extract_data(df_test))
    else:
        data_train = extract_data(df_train)
        data_test = extract_data(df_test)
    
    return data_train, data_test

    
# Recibe la matriz de datos y realiza un análisis de como son los autovalores.
def autoval_analysis(data):
    data = data - np.mean(data) # Datos originales insesgados
    cov = np.matmul(data.T, data) # Matriz de covarianzas
    
    _, s_data, _ = svd(data)
    plt.yscale("log")
    plt.plot((s_data))
    plt.title("Descenso de Autovalores en la matriz de datos")
    plt.show()
    
    plt.yscale("log")
    plt.xscale("log")
    plt.plot((s_data))
    plt.title("Descenso de Autovalores en la matriz de datos (logarítmico)")
    plt.show()
    
    _, s_cov, _ = svd(cov)
    Total = sum(s_cov)
    for i in [j for j in range(1, 100, 10)] + [j for j in range(100, 1000, 100)] + [j for j in range(1000, len(s_cov), 1000)]:
        print("Tomamos %d componentes" % (i) + " y nos queda un %2.2f" % (100*(sum(s_cov[:i])/Total)) +  " % de la varianza total")
    pass
   
    
def pca_metric_test(data_train, data_test, proves):
    start = timer()
    data = data_test - np.mean(data_test) # Datos originales insesgados
    cov = np.matmul(data.T, data) # Matriz de covarianzas
    
    _, s, _ = svd(cov)
    nrow, ncol = data_test.shape
    
    rang = min(nrow, ncol)
    
    for i in proves:
        rpct = i / rang
        lapstart = timer()
        
        # random_state = 1 porque los datos son mayores que 500x500 y usamos el solver "randomized" (se pone por defecto)
        pca = PCA(n_components = i, random_state = 1).fit(data_train)
        
        reduced_data = pca.transform(data_test)       
        recover = pca.inverse_transform(reduced_data)
        
        reduced_data_train = pca.transform(data_train)       
        recover_train = pca.inverse_transform(reduced_data_train)
        error_train = mean_squared_error(data_train, recover_train)
        
        error = mean_squared_error(data_test, recover)
        r2 = r2_score(data_test, recover, multioutput='variance_weighted')
        lapend = timer()

        print("N.Componentes = %d; ---> " %(i) +
              "  Varianza Retenida = %2.3f;" % (sum(s[:i])/sum(s)*100) + 
              " %" + "  MSE = %2.3f;" % (error) + 
              "   R2 = %2.3f" % (r2*100) +
              " %" + "  Reducción Total = %2.3f" % ((1 - rpct)*100) +
              " %" + " ---   Time --> %.2f" % (lapend - lapstart) + " s" + " *** Error en train: %2.3f" % (error_train))
        
    end = timer()
    print("\nTiempo --> %.2f" % (end - start) + " s")
    pass

def pca_view(data_train, data_test, ncomp):
    start = timer()
    
    nplot = 3
    n_row, n_col = 1,nplot
    nrow, ncol = data_test.shape
    
    rang = min(nrow, ncol)
    
    def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap = plt.get_cmap('YlOrRd')):
        plt.figure(figsize=(2. * n_col, 2.36 * n_row))

        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(img_shape), origin = "lower", cmap=cmap)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        
        pass
    
    rpct = ncomp / rang
    lapstart = timer()

    # random_state = 1 porque los datos son mayores que 500x500 y usamos el solver "randomized" (se pone por defecto)
    pca = PCA(n_components = ncomp, random_state = 1).fit(data_train)

    reduced_data = pca.transform(data_test)       
    recover = pca.inverse_transform(reduced_data)
   

    reduced_data_train = pca.transform(data_train)       
    recover_train = pca.inverse_transform(reduced_data_train)
    error_train = mean_squared_error(data_train, recover_train)
    
    plot_gallery("Original 2019", data_train[:nplot])
    plot_gallery("Recover 2019", recover_train[:nplot])
    
    plot_gallery("Original 2020", data_test[:nplot])
    plot_gallery("Recover 2020", recover[:nplot])

    error = mean_squared_error(data_test, recover)
    r2 = r2_score(data_test, recover, multioutput='variance_weighted')
    lapend = timer()

    print("N.Componentes = %d; ---> " %(ncomp) +
          "  MSE = %2.3f;" % (error) + 
          "   R2 = %2.3f" % (r2*100) +
          " %" + "  Reducción Total = %2.3f" % ((1 - rpct)*100) +
          " %" + " ---   Time --> %.2f" % (lapend - lapstart) + " s" + " *** Error en train: %2.3f" % (error_train))
        
    end = timer()
    print("\nTiempo --> %.2f" % (end - start) + " s")
    pass
    
    
    
def pca_var_report(data, perc):
    pca = PCA(n_components =perc, svd_solver = 'full')
    reduced_data = pca.fit_transform(data)
    recover = pca.inverse_transform(reduced_data)
    error = mean_squared_error(data, recover)
    maerror = mean_absolute_error(data, recover)
    r2 = r2_score(data, recover)
    n_comp, _ = pca.components_.shape
    n_feat, _ = data.shape
    print('Testeando PCA con un porcentaje de varianza de %.3f' % (perc*100) + ' %')
    print('\nError Cuadrático Medio: %.3f %' % (error*100))
    print('R2 score: %.3f %' %(r2*100))
    print('\nNumero de componentes: '+str(n_comp)+ ', lo cual representa una reducción del %2.2f' % ((1 - n_comp/n_feat) * 100) + ' %')
    
    pass


def ica_metric_test(data_train, data_test, proves):
    start = timer()
    
    for i in proves:
        lapstart = timer()
        
        # random_state = 1 porque los datos son mayores que 500x500 y usamos el solver "randomized" (se pone por defecto)
        ica = FastICA(n_components=i, whiten=True).fit(data_train)
        
        reduced_data = ica.transform(data_test)       
        recover = ica.inverse_transform(reduced_data)
        
        reduced_data_train = ica.transform(data_train)       
        recover_train = ica.inverse_transform(reduced_data_train)
        error_train = mean_squared_error(data_train, recover_train)
        
        error = mean_squared_error(data_test, recover)
        r2 = r2_score(data_test, recover, multioutput='variance_weighted')
        lapend = timer()

        print("N.Componentes = %d; ---> " %(i) + 
              "  MSE = %2.3f;" % (error) + 
              "   R2 = %2.3f" % (r2*100) +
              " %" + " ---   Time --> %.2f" % (lapend - lapstart) + " s" + " *** Error en train: %2.3f" % (error_train))
        
    end = timer()
    print("\nTiempo --> %.2f" % (end - start) + " s")
    pass
    