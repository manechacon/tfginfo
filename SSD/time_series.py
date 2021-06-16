import ssd_utils
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import time
import joblib
import pandas as pd
from scipy.linalg import svd
import os

os.environ['OMP_NUM_THREADS'] = '3'

# Preprocesamiento de datos

data_orig_2019 = ssd_utils.extract_data(ssd_utils.read_data_2019())

data_fwd_2019 = data_orig_2019[1:, :]
data_bck_2019 = data_orig_2019[:-1, :]
# data_fwd_2020 = data_orig_2020[1:, :]
# data_bck_2020 = data_orig_2020[:-1, :]


# Modelo lineal

n_folds = 2
kf = KFold(n_folds, shuffle=False)

clf = Ridge(alpha = 10.**-4)


regr = Pipeline(steps=[('std_sc', StandardScaler()),
                       ('ridge', clf)])

y_transformer = StandardScaler()
inner_estimator = TransformedTargetRegressor(regressor=regr,
                                             transformer=y_transformer)

l_alpha = [10.**k for k in range(-10, 10)]
param_grid = {'regressor__ridge__alpha': l_alpha}  

cv_estimator = GridSearchCV(inner_estimator, 
                            param_grid=param_grid, 
                            cv=kf, 
                            scoring='neg_mean_squared_error',
                            return_train_score=True,
                            refit=True,
                            n_jobs=5, 
                            verbose=1)

t_0 = time.time()
cv_estimator.fit(data_bck_2019, data_fwd_2019)
t_1 = time.time() 
print("\nridge_grid_search_time: %.2f" % ((t_1 - t_0)/60.))
        
# saving alpha_search in a pickle    
f_name = 'ts_results_ridge_2019.joblib'
joblib.dump(cv_estimator, f_name, compress=3)


# Modelo Neuronal

n_folds = 2
kf = KFold(n_folds, shuffle=False)


mlp = MLPRegressor(hidden_layer_sizes = (10**3, 10**3), 
                            solver = 'adam',
                            activation = 'relu', 
                            random_state=1, 
                            tol = 1.e-6, 
                            alpha = 10.**-2,
                            n_iter_no_change = 100,
                            max_iter=5000, 
                            shuffle = True,
                            early_stopping = False,
                            verbose = True)


regr = Pipeline(steps=[('std_sc', StandardScaler()),
                       ('mlp', mlp)])

y_transformer = StandardScaler()
inner_estimator = TransformedTargetRegressor(regressor=regr,
                                             transformer=y_transformer)

l_alpha = [10.**k for k in range(-4, 4)]
param_grid = {'regressor__mlp__alpha': l_alpha}   

cv_estimator = GridSearchCV(inner_estimator, 
                            param_grid=param_grid, 
                            cv=kf, 
                            scoring='neg_mean_squared_error',
                            return_train_score=True,
                            refit=True,
                            n_jobs=5, 
                            verbose=1)

t_0 = time.time()
cv_estimator.fit(data_bck_2019, data_fwd_2019)
t_1 = time.time() 
print("\nmlp_grid_search_time: %.2f" % ((t_1 - t_0)/60.))
        
# saving alpha_search in a pickle    
f_name = 'stored_objs/ts_results_mlp_2019_v3.joblib'
joblib.dump(cv_estimator, f_name, compress=3)