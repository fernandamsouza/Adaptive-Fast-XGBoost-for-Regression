import argumentos
import numpy as np
import salvar_resultados
from data_stream_generators import get_dataset

from adaptive_xgboost import AdaptiveXGBoostClassifier
from adaptive_semiV2 import AdaptiveSemi
from modelos_adaptados_para_sklearn import (
    AdaptiveRandomForestClassifierA,
    HoeffdingAdaptiveTreeClassifierA,
    AdaptiveSemiRegressorJr2,
    AdaptiveSemiRegressorJ2
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
#define your own mse and set greater_is_better=False
mse = make_scorer(mean_squared_error,greater_is_better=False)

def MSE(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print ('MSE: %2.3f' % mse)
    return mse

def R2(y_true,y_pred):    
     r2 = r2_score(y_true, y_pred)
     print ('R2: %2.3f' % r2)
     return r2

def two_score(y_true,y_pred):    
    MSE(y_true,y_pred) #set score here and not below if using MSE in GridCV
    score = R2(y_true,y_pred)
    return score

def two_scorer():
    return make_scorer(two_score, greater_is_better=False) # change for false if using MSE


def _criar_modelo(**kwargs):
    if argumentos.CLASSIFICADOR == "axgb":
        return AdaptiveXGBoostClassifier(**kwargs)
    elif argumentos.CLASSIFICADOR == "axgb_reset":
        return AdaptiveSemiRegressorJr2(detect_drift=True, unic="N", **kwargs)
    elif argumentos.CLASSIFICADOR == "axgb_sem_reset":
        return AdaptiveSemiRegressorJ2(detect_drift=True, unic="N", **kwargs)


parameter_grid = {}
parameter_grid = {
    "max_depth": [1, 5, 10, 15],
    "learning_rate": [0.01, 0.05, 0.1, 0.5],
    "max_window_size": [512, 1024, 2048, 4096, 8192],
    "min_window_size": [4, 8, 16],
}

parameter_grid_drifts = {
    "adwin_delta": [1, 0.002, 0.003, 0.00001, 0.0001],
    "kswin_alpha": [1, 0.005, 0.003, 0.00001, 0.0001],
    "kswin_window_size": [100, 500],
    "kswin_stat_size": [30, 100],
    "ddm_min_num_instances": [30, 50],
    "ddm_warning_level": [1, 2],
    "ddm_out_control_level": [3, 5]
}

print(f"Carregando dataset {argumentos.DATASET}")
dataset = np.loadtxt(f"datasets/{argumentos.DATASET}.csv", delimiter=",", skiprows=1)
print(f"Configurando dataset")

print(dataset)
X, y = (
    dataset[: argumentos.MAX_REGISTROS, :-1],
    dataset[: argumentos.MAX_REGISTROS, -1],
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.3, random_state=1
)

print(X_train, X_test,y_train, y_test)
gs_cv = GridSearchCV(_criar_modelo(), parameter_grid_drifts, scoring=two_scorer(), n_jobs=2)

print(f"Realizando GridSearchCV")
result = gs_cv.fit(X_train, y_train)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

print("Salvando resultados")
salvar_resultados.salvar_resultados_gridsearch(gs_cv.cv_results_, gs_cv.best_params_)
