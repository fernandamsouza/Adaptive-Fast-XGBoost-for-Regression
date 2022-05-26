from turtle import ht
from scipy import rand
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from adaptive_xgboost import AdaptiveXGBoostClassifier
# from adaptive_incremental_ensemble import AdaptiveXGBoostClassifier2
# from adaptive_incremental import Adaptive2
from adaptive_xgboost_thread import Adaptive3
# from adaptive_incremental2 import Adaptive4
from adaptive_semiV2 import AdaptiveSemi
from adaptive_xgboost_2 import AdaptiveXGBoostClassifier2
from adaptive_semiV3r_regressor import AdaptiveSemiRegressorJr
from adaptive_semiV3_regressor import AdaptiveSemiRegressorJ
from adaptive_semiV3r_regressor_JULIA import AdaptiveSemiRegressorJULIA
from adaptive_semiV3_regressor_S import AdaptiveSemiRegressorJS


from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.trees import iSOUPTreeRegressor
from skmultiflow.trees import StackedSingleTargetHoeffdingTreeRegressor
from skmultiflow.lazy import KNNRegressor

from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.data import SEAGenerator, RegressionGenerator, LEDGenerator, RandomRBFGenerator, HyperplaneGenerator, WaveformGenerator, RandomTreeGenerator, MIXEDGenerator


# Adaptive XGBoost classifier parameters
# n_estimators = 30       # Number of members in the ensemble
# learning_rate = 0.3     # Learning rate or eta
# max_depth = 6           # Max depth for each tree in the ensemble
# max_window_size = 1000  # Max window size
# min_window_size = 1     # set to activate the dynamic window strategy
# detect_drift = True    # Enable/disable drift detection
# ratio_unsampled = 0
# small_window_size = 150

# max_buffer = 50
# pre_train = 20

n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.3     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 1000  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = True    # Enable/disable drift detection
ratio_unsampled = 0
small_window_size = 150
max_samples = 500000
width = max_samples * 0.02
pre_train = 15


max_window = 500
max_buffer=25
pre_training=15
## autor push
# AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
#                                   n_estimators=n_estimators,
#                                   learning_rate=learning_rate,
#                                   max_depth=max_depth,
#                                   max_window_size=max_window_size,
#                                   min_window_size=min_window_size,
#                                   detect_drift=detect_drift)
## meu ensemble incremental
# AXGBp2 = AdaptiveXGBoostClassifier2(update_strategy='push',
#                                   n_estimators=n_estimators,
#                                   learning_rate=learning_rate,
#                                   max_depth=max_depth,
#                                   max_window_size=max_window_size,
#                                   min_window_size=min_window_size,
#                                   detect_drift=detect_drift)
## autor replace
# AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
#                                   n_estimators=n_estimators,
#                                   learning_rate=learning_rate,
#                                   max_depth=max_depth,
#                                   max_window_size=max_window_size,
#                                   min_window_size=min_window_size,
#                                   detect_drift=detect_drift,
#                                   ratio_unsampled=ratio_unsampled)

## meu incremental
# AXGBg = Adaptive4(learning_rate=learning_rate,
#                                   max_depth=max_depth,
#                                   max_window_size=max_window_size,
#                                   min_window_size=min_window_size,
#                                   ratio_unsampled=ratio_unsampled)

AXGBg2 = AdaptiveSemi(learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  ratio_unsampled=ratio_unsampled,
                                  small_window_size=small_window_size,
                                  max_buffer=max_buffer,
                                  pre_train=pre_train,
                                  detect_drift=False,
                                  unic = "N")

AXGBg3 = AdaptiveSemi(learning_rate=learning_rate,
                                max_depth=max_depth,
                                max_window_size=max_window_size,
                                min_window_size=min_window_size,
                                ratio_unsampled=ratio_unsampled,
                                small_window_size=small_window_size,
                                max_buffer=max_buffer,
                                pre_train=pre_train,
                                detect_drift=True,
                                unic = "N")

AXGBr = AdaptiveXGBoostClassifier2(update_strategy='replace',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=False)

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

# ## meu thread
# AXGBt = Adaptive3(n_estimators=n_estimators,
#                                   learning_rate=learning_rate,
#                                   max_depth=max_depth,
#                                   max_window_size=max_window_size,
#                                   min_window_size=min_window_size,
#                                   detect_drift=detect_drift)

# stream = FileStream("streaming-datasets-master/airlines.csv")
# stream = SEAGenerator(noise_percentage=0.1)
# stream = HyperplaneGenerator()
# stream = RegressionGenerator(n_samples=50000)


# abalone = FileStream("datasets/abalone.csv")
# ailerons = FileStream("datasets/ailerons.csv")
# bike = FileStream("datasets/bikes_clean.csv")
# fried_delve = FileStream("datasets/fried_delve.csv")
# elevators = FileStream("datasets/elevators.csv")
# house8l = FileStream("datasets/house8L.csv")
# house16h = FileStream("datasets/house16H.csv")
# cal_housing = FileStream("datasets/cal_housing.csv")
# pol = FileStream("datasets/pol.csv")
# spat_network_3d = FileStream("datasets/3D_spatial_network.csv")
# metrotraffic = FileStream("datasets/Metro_Interstate_Traffic_Volume_clean.csv")

# stream = pol
# stream = ConceptDriftStream(random_state=112, position=10000, width=1)
# stream = ConceptDriftStream(random_state=1000, position=5000)
# stream = ConceptDriftStream(random_state=112)
# stream = AGRAWALGenerator()
# stream = HyperplaneGenerator()

reg1 = ConceptDriftStream(stream=RegressionGenerator(n_samples=500000, n_features=10, random_state=1),drift_stream=RegressionGenerator(n_samples=500000, n_features=10, random_state=2), position = max_samples/4, width = 1)
reg2 = ConceptDriftStream(stream=reg1, drift_stream=RegressionGenerator(n_samples=500000, n_features=10, random_state=3), position = max_samples/2, width = 1)
regression_generator_drift_a4 = ConceptDriftStream(stream=reg2, drift_stream=RegressionGenerator(n_samples=500000, n_features=10, random_state=4), position = max_samples*3/4, width = 1)

# regg1 = ConceptDriftStream(stream=RegressionGenerator(n_samples=500000, n_features=10, random_state=1),drift_stream=RegressionGenerator(n_samples=500000, n_features=10, random_state=2), position = max_samples/4, width = width)
# regg2 = ConceptDriftStream(stream=regg1, drift_stream=RegressionGenerator(n_samples=500000, n_features=10, random_state=3), position = max_samples/2, width = width)
# regression_generator_drift_g4 = ConceptDriftStream(stream=regg2, drift_stream=RegressionGenerator(n_samples=500000, n_features=10, random_state=4), position = max_samples*3/4, width = width)

stream = regression_generator_drift_a4

HTR = HoeffdingTreeRegressor()
# HTRA = HoeffdingAdaptiveTreeRegressor()
# ISOUP = iSOUPTreeRegressor()
# SSHT = StackedSingleTargetHoeffdingTreeRegressor()
# KNN = KNNRegressor()

# ARFReg = AdaptiveRandomForestRegressor()

evaluator = EvaluatePrequential(pretrain_size=0,
                                max_samples=500000,
                                # batch_size=200,
                                output_file="out",
                                show_plot=True,
                                metrics=["mean_square_error", "running_time"])

# evaluator.evaluate(stream=stream,
#                    model=[AXGBRegRD, AXGBRegR, AXGBRegSD, AXGBRegS, HTR],
#                    model_names=["D+RESET", "RESET", "D", "D_SEM_RESET", "HTR"])
evaluator.evaluate(stream=stream,
model=[AXGBRegS],
model_names=["D_SEM_RESET"])

# evaluator.evaluate(stream=stream,
#                    model=[AXGBRegSD],
#                    model_names=["D"])

# evaluator.evaluate(stream=stream,
#                    model=[AXGBg3],
#                    model_names=["AXGBg3"])
