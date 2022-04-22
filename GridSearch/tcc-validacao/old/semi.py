from skmultiflow.data import SEAGenerator
from skmultiflow.meta import BatchIncrementalClassifier
from adaptive_semi import AdaptiveSemi
from skmultiflow.data.file_stream import FileStream
# Setup a data stream
stream = FileStream("./datasets/australian.csv")

# Pre-training the classifier with 200 samples
X, y = stream.next_sample(100)
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.2     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 50  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = False    # Enable/disable drift detection
ratio_unsampled = 0.99
small_window_size = 100

AXGBg2 = AdaptiveSemi(learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  ratio_unsampled=ratio_unsampled,
                                  small_window_size=small_window_size)

AXGBg2.partial_fit(X, y)

# Preparing the processing of 5000 samples and correct prediction count
n_samples = 0
correct_cnt = 0
while n_samples < 600 and stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = AXGBg2.predict(X)
    if y[0] == y_pred[0]:
        correct_cnt += 1
    AXGBg2.partial_fit(X, y)
    n_samples += 1

# Display results
print('Batch Incremental ensemble classifier example')
print('{} samples analyzed'.format(n_samples))
print('Performance: {}'.format(correct_cnt / n_samples))