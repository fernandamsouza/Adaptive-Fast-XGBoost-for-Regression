from adaptive_xgboost import AdaptiveXGBoostClassifier
# from adaptive_incremental_ensemble import AdaptiveXGBoostClassifier2
# from adaptive_incremental import Adaptive2
# from adaptive_xgboost_thread import Adaptive3
from adaptive_semiV2 import AdaptiveSemi
# from adaptive_incremental2_adwin import Adaptive5
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.hyper_plane_generator  import HyperplaneGenerator
from skmultiflow.meta import OnlineBoostingClassifier

from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream

# Adaptive XGBoost classifier parameters
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.05     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 10000  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = False    # Enable/disable drift detection
ratio_unsampled = 0
small_window_size = 150
par_boost = [512,1024,2048,4096,8192]
classificadores = []
labels = []

max_buffer = 50
pre_train = 20

# for x in par_boost:
# ## autor replace
#     # meu incremental
AXGBg2 = AdaptiveSemi(learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  ratio_unsampled=ratio_unsampled,
                                  small_window_size=small_window_size,
                                  max_buffer=max_buffer,
                                  pre_train=pre_train)
AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
                                n_estimators=30,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                ratio_unsampled=ratio_unsampled,
                                max_window_size=max_window_size,
                                min_window_size=min_window_size,
                                detect_drift=detect_drift)
# online_boosting = OnlineBoostingClassifier()
labels.append('AXGB adaptado')
classificadores.append(AXGBg2)

# AXGBg2 = Adaptive5(learning_rate=learning_rate,
#                                 max_depth=max_depth,
#                                 max_window_size=max_window_size,
#                                 min_window_size=min_window_size)
# classificadores.append(AXGBg)
# labels.append('incremental')

# stream = FileStream("./datasets/hyper_f.csv")
stream = SEAGenerator(noise_percentage=0.1)
# stream.prepare_for_use()   # Required for skmultiflow v0.4.1

# print(labels)
evaluator = EvaluatePrequential(pretrain_size=0,
                                max_samples=200000,
                                # batch_size=5000,
                                show_plot=False,
                                metrics=["accuracy","running_time"])

evaluator.evaluate(stream=stream,
                   model=classificadores,
                   model_names=labels)
