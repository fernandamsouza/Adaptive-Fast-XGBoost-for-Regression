from turtle import ht
from scipy import rand
import argumentos

from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from adaptive_reset_array_regressor import AdaptiveSemiRegressorJr
from adaptive_array_regressor import AdaptiveSemiRegressorJ
from adaptive_reset_regressor import AdaptiveSemiRegressorJULIA
from adaptive_regressor import AdaptiveSemiRegressorJS


from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential
 
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.trees import iSOUPTreeRegressor
from skmultiflow.trees import StackedSingleTargetHoeffdingTreeRegressor
from skmultiflow.lazy import KNNRegressor

from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.data import SEAGenerator, RegressionGenerator, LEDGenerator, RandomRBFGenerator, HyperplaneGenerator, WaveformGenerator, RandomTreeGenerator, MIXEDGenerator

n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.3     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 1000  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = True    # Enable/disable drift detection
ratio_unsampled = 0
small_window_size = 150
max_samples = argumentos.MAX_REGISTROS
width = max_samples * 0.02
pre_train = 15

max_buffer=25
pre_training=15


# MODELOS DA JULIA - REGRESSÃO

AXGBRegRD = AdaptiveSemiRegressorJr(learning_rate=learning_rate,
                                max_depth=max_depth,
                                max_window_size=max_window_size,
                                min_window_size=min_window_size,
                                ratio_unsampled=ratio_unsampled,
                                small_window_size=small_window_size,
                                max_buffer=max_buffer,
                                pre_train=pre_train,
                                detect_drift=True,
                                unic = "N")

AXGBRegR = AdaptiveSemiRegressorJULIA(learning_rate=learning_rate,
                                max_depth=max_depth,
                                max_window_size=max_window_size,
                                min_window_size=min_window_size,
                                ratio_unsampled=ratio_unsampled,
                                small_window_size=small_window_size,
                                max_buffer=max_buffer,
                                pre_train=pre_train)

AXGBRegSD = AdaptiveSemiRegressorJ(learning_rate=learning_rate,
                                max_depth=max_depth,
                                max_window_size=max_window_size,
                                min_window_size=min_window_size,
                                ratio_unsampled=ratio_unsampled,
                                small_window_size=small_window_size,
                                max_buffer=max_buffer,
                                pre_train=pre_train,
                                detect_drift=True,
                                unic = "N")

AXGBRegS = AdaptiveSemiRegressorJS(learning_rate=learning_rate,
                                max_depth=max_depth,
                                max_window_size=max_window_size,
                                min_window_size=min_window_size,
                                ratio_unsampled=ratio_unsampled,
                                small_window_size=small_window_size,
                                max_buffer=max_buffer,
                                pre_train=pre_train)

hyperplane = HyperplaneGenerator()

reg1 = ConceptDriftStream(stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=1),drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=2), position = 50000, width = 1)
reg2 = ConceptDriftStream(stream=reg1, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=3), position = 100000, width = 1)
reg3 = ConceptDriftStream(stream=reg2, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=3), position = 150000, width = 1)
reg4 = ConceptDriftStream(stream=reg3, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=3), position = 200000, width = 1)
reg5 = ConceptDriftStream(stream=reg4, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=3), position = 250000, width = 1)
reg6 = ConceptDriftStream(stream=reg5, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=3), position = 300000, width = 1)
reg7 = ConceptDriftStream(stream=reg6, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=3), position = 350000, width = 1)
reg8 = ConceptDriftStream(stream=reg7, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=3), position = 400000, width = 1)
regression_generator_drift_a4 = ConceptDriftStream(stream=reg8, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=4), position = 450000, width = 1)

regg1 = ConceptDriftStream(stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=1),drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=2), position = max_samples/4, width = width)
regg2 = ConceptDriftStream(stream=regg1, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=3), position = max_samples/2, width = width)
regression_generator_drift_g4 = ConceptDriftStream(stream=regg2, drift_stream=RegressionGenerator(n_samples=argumentos.MAX_REGISTROS, n_features=10, random_state=4), position = max_samples*3/4, width = width)

# stream = regression_generator_drift_a4

HTR = HoeffdingTreeRegressor()
HTRA = HoeffdingAdaptiveTreeRegressor()
KNN = KNNRegressor()

ARFReg = AdaptiveRandomForestRegressor()

# Criar Stream
stream = None
if argumentos.DATASET == "abrupto":
    stream = regression_generator_drift_a4
elif argumentos.DATASET == "gradual":
    stream = regression_generator_drift_g4
elif argumentos.DATASET == "incremental":
    stream = hyperplane
# elif argumentos.DATASET == "real":
#     stream = fried_delve
    
# Criar modelo
model = None
if argumentos.CLASSIFICADOR == "AXGBRegRD":
    model = AXGBRegRD
elif argumentos.CLASSIFICADOR == "AXGBRegR":
    model = AXGBRegR
elif argumentos.CLASSIFICADOR == "AXGBRegSD":
    model = AXGBRegSD
elif argumentos.CLASSIFICADOR == "AXGBRegS":
    model = AXGBRegS
elif argumentos.CLASSIFICADOR == "HTR":
    model = HTR
elif argumentos.CLASSIFICADOR == "KNN":
    model = KNN
elif argumentos.CLASSIFICADOR == "HTRA":
    model = HTRA
elif argumentos.CLASSIFICADOR == "ARFReg":
    model = ARFReg

evaluator = EvaluatePrequential(pretrain_size=0,
                                max_samples=argumentos.MAX_REGISTROS,
                                # batch_size=200,
                                output_file="resultados_script_final/exec_" + str(argumentos.CLASSIFICADOR) + "_" + str(argumentos.DATASET) + str(argumentos.ITERACAO) + ".out",
                                show_plot=True,
                                metrics=["mean_square_error", "running_time"])

# evaluator.evaluate(
#     stream=stream,
#     model=model,
#     model_names=[argumentos.CLASSIFICADOR],
# )
evaluator.evaluate(stream=stream,
                   model=[AXGBRegRD, AXGBRegR, AXGBRegSD, AXGBRegS, HTR, HTRA],
                   model_names=["D+RESET", "RESET", "D-SEM", "SEM_RESET", "HTR", "HTRA"])


# evaluator.evaluate(stream=stream,
# model=[AXGBRegS],
# model_names=["D_SEM_RESET"])

# evaluator.evaluate(stream=stream,
#                    model=[AXGBRegSD],
#                    model_names=["D"])

# evaluator.evaluate(stream=stream,
#                    model=[AXGBg3],
#                    model_names=["AXGBg3"])
