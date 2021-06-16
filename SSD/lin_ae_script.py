import ssd_utils
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import time
import joblib

data2019 = ssd_utils.get_all_data_2019(norm = False, itp = True)
data2020 = ssd_utils.get_all_data_2020(norm = False, itp = True)


n_folds = 2
kf = KFold(n_folds, shuffle=False)

mlp = MLPRegressor(hidden_layer_sizes = (300), 
                            solver = 'adam',
                            activation = 'identity', 
                            random_state=1, 
                            tol = 1.e-6, 
                            n_iter_no_change = 100,
                            max_iter=1000, 
                            shuffle = True,
                            early_stopping = False,
                            verbose = True)

regr = Pipeline(steps=[('std_sc', StandardScaler()),
                       ('mlp', mlp)])

y_transformer = StandardScaler()
inner_estimator = TransformedTargetRegressor(regressor=regr,
                                             transformer=y_transformer)

l_alpha = [10.**k for k in range(-5, 5)]
param_grid = {'regressor__mlp__alpha': l_alpha}  

cv_estimator = GridSearchCV(inner_estimator, 
                            param_grid=param_grid, 
                            cv=kf, 
                            scoring='neg_mean_squared_error',
                            return_train_score=True,
                            refit=True,
                            n_jobs=2, 
                            verbose=1)
t_0 = time.time()
cv_estimator.fit(data2019, data2019)
t_1 = time.time() 
print("\nmlp_grid_search_time: %.2f" % ((t_1 - t_0)/60.))
        
# saving alpha_search in a pickle    
f_name = 'stored_objs/linear(300)_100patience.joblib'
joblib.dump(cv_estimator, f_name, compress=3)