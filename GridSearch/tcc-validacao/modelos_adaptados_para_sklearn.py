# Esses modelos foram adaptados para que tenham compatibilidade com GridSearchCV do sklearn.
# Sem essas adaptações, o GridSearchCV dará erro

from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from adaptive_semiV3r_regressor import AdaptiveSemiRegressorJr
from adaptive_semiV3_regressor import AdaptiveSemiRegressorJ


class HoeffdingAdaptiveTreeClassifierA(HoeffdingAdaptiveTreeClassifier):
    def _more_tags(self):
        return {"pairwise": False}


class AdaptiveRandomForestClassifierA(AdaptiveRandomForestClassifier):
    def _more_tags(self):
        return {"pairwise": False}

class AdaptiveSemiRegressorJr2(AdaptiveSemiRegressorJr):
    def _get_tags(self):
        return {"pairwise": False}

class AdaptiveSemiRegressorJ2(AdaptiveSemiRegressorJ):
    def _get_tags(self):
        return {"pairwise": False}