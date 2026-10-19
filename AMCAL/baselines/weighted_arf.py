"""Sample-weight adapter for River 0.26.1 ARF, including background trees.

River's ARF.learn_one accepts **kwargs but does not propagate an outer sample
weight. Keep its voting, Poisson(6), warnings, drift and replacement machinery;
multiply tree-level bagging weights by the paid sample's MicFoal weight exactly
once. No rounding/repeated calls and no second Poisson draw are introduced.
"""
import math
from river import forest


class WeightedTree:
    def __init__(self, tree, weight_box):
        self.tree = tree
        self.weight_box = weight_box

    def learn_one(self, x, y, w=1, **kwargs):
        return self.tree.learn_one(x, y, w=w*self.weight_box[0], **kwargs)

    def predict_one(self, x):
        return self.tree.predict_one(x)

    def predict_proba_one(self, x):
        return self.tree.predict_proba_one(x)


class WeightedARFClassifier(forest.ARFClassifier):
    def _new_base_model(self):
        if not hasattr(self, '_sample_weight_box'):
            self._sample_weight_box = [1.]
        return WeightedTree(super()._new_base_model(), self._sample_weight_box)

    def learn_one(self, x, y, w=1., **kwargs):
        if not math.isfinite(w) or w <= 0:
            raise ValueError('ARF sample weight must be finite and positive.')
        if not hasattr(self, '_sample_weight_box'):
            self._sample_weight_box = [1.]
        self._sample_weight_box[0] = w
        try:
            return super().learn_one(x, y, **kwargs)
        finally:
            self._sample_weight_box[0] = 1.
