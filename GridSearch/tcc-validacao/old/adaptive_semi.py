import numpy as np

import xgboost as xgb
import random

from skmultiflow.core.base import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
from sklearn.neighbors import KNeighborsClassifier


class AdaptiveSemi(BaseSKMObject, ClassifierMixin):

    def __init__(self,
                 learning_rate=0.3,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None,
                 ratio_unsampled=0,
                 small_window_size=0):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self._first_run = True
        self._booster = None
        self._drift_detector = None
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])

        self._ratio_unsampled = ratio_unsampled
        self._X_small_buffer = np.array([])
        self._y_small_buffer = np.array([])
        self._samples_seen = 0
        self._model_idx = 0
        self._small_window_size = small_window_size

        self._configure()

    def _configure(self):
        self._reset_window_size()
        self._init_margin = 0.0
        self._boosting_params = {
            "objective": "binary:logistic",
            "eta": self.learning_rate,
            'update':'refresh',
            'process_type': 'default',
            'refresh_leaf': True,
            "n_estimators": 100,
            "num_parallel_tree": 1,
            "eval_metric": "logloss",
            "max_depth": self.max_depth}

    def reset(self):
        self._first_run = True
        self._configure()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Partially (incrementally) fit the model.

        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the data upon which
            the algorithm will create its model.

        y: Array-like
            An array of shape (, n_samples) containing the classification
            targets for all samples in X. Only binary data is supported.

        classes: Not used.

        sample_weight: Not used.

        Returns
        -------
        AdaptiveXGBoostClassifier
            self
        """
        for i in range(X.shape[0]):
            self._partial_fit(np.array([X[i, :]]), np.array([y[i]]))
        return self

    def _change_small_window(self, npArrX, npArrY):
        if npArrX.shape[0] < self._small_window_size:
            sizeToRemove = 0
            nextSize = self._X_small_buffer.shape[0] + npArrX.shape[0]
            if nextSize > self._small_window_size:
                sizeToRemove = nextSize - self._small_window_size
            #deleta os dados velhos
            delete_idx = [i for i in range(sizeToRemove)]

            if len(delete_idx) > 0:
                self._X_small_buffer = np.delete(self._X_small_buffer, delete_idx, axis=0)
                self._y_small_buffer = np.delete(self._y_small_buffer, delete_idx, axis=0)
            
            self._X_small_buffer = np.concatenate((self._X_small_buffer, npArrX))
            self._y_small_buffer = np.concatenate((self._y_small_buffer, npArrY))
        else:
            self._X_small_buffer = npArrX[0:self._small_window_size]
            self._y_small_buffer = npArrY[0:self._small_window_size]

    def _unlabeled_fit(self):
        # unlabeled = map(lambda x: x != 0 and x != 1, self._y_buffer)
        npArrX = []
        npArrY = []

        unlabeled = []
        labeledX = []
        labeledY = []
        for i in range(len(self._X_buffer)):
            currentY = self._y_buffer[i]

            max_size = int(self._ratio_unsampled * len(self._X_buffer))
            # print(max_size)
            if max_size > i:
                unlabeled.append(self._X_buffer[i])
            else:
                labeledX.append(self._X_buffer[i])
                labeledY.append(currentY)
            # if currentY != 1 and currentY != 0:
            #     unlabeled.append(self._X_buffer[i])
            # else:
            #     labeledX.append(self._X_buffer[i])
            #     labeledY.append(currentY)
        npArrX = np.array(labeledX)
        npArrY = np.array(labeledY)
        if npArrX.shape[0] > 0:
            self._change_small_window(npArrX, npArrY)
        npUnlabeled = np.array(unlabeled)

        if npArrX.shape[0] > 6:
            if npUnlabeled.shape[0] > 0:
                nbrs = KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree').fit(self._X_small_buffer, self._y_small_buffer)

                proba = nbrs.predict_proba(npUnlabeled)

                for j in range(len(proba)):
                    biggerIndex = np.argmax(proba[j])
                    otherIndex = biggerIndex == 0 and 1 or 0
                    margim = proba[j][biggerIndex] - proba[j][otherIndex]

                    if (margim > 0.5):
                        # print("proba")
                        # print(proba[j])
                        # print("incerteza")
                        # print(margim)
                        # print("y q eu acho")
                        # print(biggerIndex)
                        # print("y certo")
                        # print(npArrY[j])
                        npArrXNew = np.array([npUnlabeled[j]])
                        npArrYNew = np.array([biggerIndex])
                        npArrX = np.concatenate((npArrX, npArrXNew))
                        npArrY = np.concatenate((npArrY, npArrYNew))
        # print("semi")
        # print(len(npArrX))
        return (npArrX, npArrY)


    def _partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_buffer = np.array([])
            self._X_small_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_small_buffer = np.array([])
            self._first_run = False
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))

        while self._X_buffer.shape[0] >= self.window_size:
            
            npArrX, npArrY = self._unlabeled_fit()
            if npArrX.shape[0] > 0:
                self._train_on_mini_batch(X=npArrX,
                                        y=npArrY)
            delete_idx = [i for i in range(self.window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)

            # Check window size and adjust it if necessary
            self._adjust_window_size()

    def _adjust_window_size(self):
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self.window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self.window_size = self._dynamic_window_size

    def _train_on_mini_batch(self, X, y):
        booster = self._train_booster(X, y, self._model_idx)
        # Update ensemble
        self._booster = booster

    def _train_booster(self, X: np.ndarray, y: np.ndarray, last_model_idx: int):
        d_mini_batch_train = xgb.DMatrix(X, y.astype(int))

        if self._booster:
            booster = xgb.train(params=self._boosting_params,
                                dtrain=d_mini_batch_train,
                                num_boost_round=1,
                                xgb_model='model_1.model')
            # booster.save_model('model_1.model')
        else:
            booster = xgb.train(params=self._boosting_params,
                                dtrain=d_mini_batch_train,
                                num_boost_round=1,
                                verbose_eval=False)
            # booster.save_model('model_1.model')
        return booster

    def predict(self, X):
        """
        Predict the class label for sample X

        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the samples to
            predict the class label for.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (, n_samples), containing the
            predicted class labels for all instances in X.

        """
        # start_time = time.time()
        if self._booster:
            d_test = xgb.DMatrix(X)
            predicted = self._booster.predict(d_test)
            return np.array(predicted > 0.5).astype(int)
        # Ensemble is empty, return default values (0)
        return np.zeros(get_dimensions(X)[0])

    def predict_proba(self, X):
        """
        Not implemented for this method.
        """
        raise NotImplementedError(
            "predict_proba is not implemented for this method.")
