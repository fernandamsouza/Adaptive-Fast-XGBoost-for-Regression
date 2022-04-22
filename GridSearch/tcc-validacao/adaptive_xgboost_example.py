import salvar_resultados
import argumentos

from adaptive_xgboost import AdaptiveXGBoostClassifier
from adaptive_semiV2 import AdaptiveSemi
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
from data_stream_generators import get_dataset


# # Adaptive XGBoost classifier parameters
# n_estimators = 30  # Number of members in the ensemble
# learning_rate = (
#     0.05
#     if not argumentos.HYPERPARAMETRO == "learning_rate"
#     else float(argumentos.VALOR_HYPERPARAMETRO)
# )  # Learning rate or eta
# max_depth = (
#     5
#     if not argumentos.HYPERPARAMETRO == "max_depth"
#     else int(argumentos.VALOR_HYPERPARAMETRO)
# )  # Max depth for each tree in the ensemble
# max_window_size = (
#     10000
#     if not argumentos.HYPERPARAMETRO == "max_window_size"
#     else int(argumentos.VALOR_HYPERPARAMETRO)
# )  # Max window size
# min_window_size = 1  # set to activate the dynamic window strategy
# detect_drift = False  # Enable/disable drift detection
# ratio_unsampled = 0
# small_window_size = 150

# max_buffer = 25
# pre_train = 15

# Criar fluxo de dados
# dataset = np.loadtxt(f"datasets/{argumentos.DATASET}.csv", delimiter=",", skiprows=1)
# X, y = dataset[:, :-1], dataset[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=0.3, random_state=1
# )
# stream = DataStream(X_test, y_test, name=f"{argumentos.DATASET}.csv")
# stream = FileStream(f"datasets/{argumentos.DATASET}.csv")
stream = get_dataset(argumentos.DATASET, argumentos.MAX_REGISTROS)


# Criar modelo
model = None
if argumentos.CLASSIFICADOR == "axgb":
    model = AdaptiveXGBoostClassifier(**argumentos.HIPER_PARAMETROS)
elif argumentos.CLASSIFICADOR == "incremental":
    model = AdaptiveSemi(**argumentos.HIPER_PARAMETROS)
elif argumentos.CLASSIFICADOR == "arf":
    model = AdaptiveRandomForestClassifier(**argumentos.HIPER_PARAMETROS)
elif argumentos.CLASSIFICADOR == "hat":
    model = HoeffdingAdaptiveTreeClassifier(**argumentos.HIPER_PARAMETROS)

evaluator = EvaluatePrequential(
    pretrain_size=0,
    max_samples=argumentos.MAX_REGISTROS,
    # batch_size=200,
    output_file=salvar_resultados.caminho_resultado_raw,
    show_plot=False,
    metrics=["accuracy", "running_time", "kappa"],
)

evaluator.evaluate(
    stream=stream,
    model=model,
    model_names=[argumentos.CLASSIFICADOR],
)

salvar_resultados.salvar_resultados_normal()
