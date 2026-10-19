"""RAL classes extracted unchanged from AMCAL/ANT_Burst_TEST/RAL.ipynb.
The shared runner enforces the global cap outside the original prefix gate.
"""
import random
from copy import deepcopy
import numpy as np
from sklearn.ensemble import VotingClassifier

class TrainingData:
    def __init__(self, X, y):
        self._X = deepcopy(X)
        self._y = deepcopy(y)

    def append(self, x, y):
        self._X.append(x)
        self._y.append(y)

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

class RAL:
    def __init__(self, committee_classifiers, threshold_uncertainty, threshold_greedy, eta, reward=1, penalty=-1,
                 budget=None, model_update_frequency=1):
        self._committee_classifiers = deepcopy(committee_classifiers)
        self._threshold_uncertainty = threshold_uncertainty
        self._threshold_greedy = threshold_greedy
        self._eta = eta
        self._reward = reward
        self._penalty = penalty
        self._budget = budget
        self._model_update_frequency = model_update_frequency

        self._committee_classifier = VotingClassifier(
            list(zip(map(str, range(len(committee_classifiers))), self._committee_classifiers)), voting='soft')

        self._alpha = [1.0 / len(self._committee_classifiers) for _ in range(len(self._committee_classifiers))]
        self._state = False
        self._n_acquired_samples = 0
        self._training_data_samples = None
        self._n_committee_too_low_acquires = 0
        self._n_committee_acquires = 0
        self._n_greedy_acquires = 0
        self._n_both_acquires = 0
        self._n_seen = 0
        self._alphas = [deepcopy(self._alpha)]

    @property
    def alphas(self):
        return self._alphas

    @property
    def n_committee_too_low_acquires(self):
        return self._n_committee_too_low_acquires

    @property
    def n_greedy_acquires(self):
        return self._n_greedy_acquires

    @property
    def n_committee_acquires(self):
        return self._n_committee_acquires

    @property
    def n_both_acquires(self):
        return self._n_both_acquires

    def fit(self, X, y):
        if self._state:
            raise AssertionError('fit has already been called.')
        self._training_data_samples = TrainingData(list(X), list(y))
        for learner in self._committee_classifiers:
            learner.fit(X, y)
        self._committee_classifier.fit(X, y)
        self._state = True

    def reward(self, y_true, y_predicted):
        if y_true != y_predicted:
            return self._reward
        else:
            return self._penalty

    def _update_alpha(self, reward, decisions, committee_decision):
        for idx, decision in enumerate(decisions):
            if int(decision) == int(committee_decision):
                self._alpha[idx] = self._alpha[idx] * np.exp(reward * self._eta)
        s = sum(self._alpha)
        for idx, coef in enumerate(self._alpha):
            self._alpha[idx] = float(coef) / s
        self._alphas.append(deepcopy(self._alpha))

    def _update_uncertainty_threshold(self, reward):
        f_reward = self._eta * (1. - np.power(2.0, float(reward) / self._penalty))
        self._threshold_uncertainty = min(self._threshold_uncertainty * (1. + f_reward), 1.)

    def _has_enough_budget(self):
        if self._budget is None:
            return True
        return float(self._n_acquired_samples) / self._n_seen < self._budget

    def ask_committee(self, x):
        decisions = []
        for learner in self._committee_classifiers:
            decisions.append(np.max(learner.predict_proba(x.reshape(1, -1))) < self._threshold_uncertainty)
        committee_decision = bool(round(sum([self._alpha[idx] * el for idx, el in enumerate(decisions)])))
        return committee_decision, decisions

    def should_label(self, committee_decision):
        self._n_seen += 1
        if not self._has_enough_budget():
            if committee_decision:
                self._n_committee_too_low_acquires += 1
            return False
        if committee_decision:
            self._n_committee_acquires += 1
        if random.random() < self._threshold_greedy:
            self._n_greedy_acquires += 1
            if committee_decision:
                self._n_both_acquires += 1
            return True
        else:
            return committee_decision

    def acquire_label(self, x, y, committee_decision, decisions):
        if committee_decision:
            yp = self._committee_classifier.predict(x.reshape(1, -1))[0]
            reward = self.reward(y, yp)
            self._update_alpha(reward, decisions, committee_decision)
            self._update_uncertainty_threshold(reward)
        self._n_acquired_samples += 1
        self._training_data_samples.append(x, y)
        if self._n_acquired_samples % self._model_update_frequency == 0:
            for learner in self._committee_classifiers:
                learner.fit(self._training_data_samples.X, self._training_data_samples.y)
            self._committee_classifier.fit(self._training_data_samples.X, self._training_data_samples.y)

    def evaluate_metrics(self, X, y):
        self._committee_classifier.fit(self._training_data_samples.X, self._training_data_samples.y)
        y_pred = self._committee_classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred
        }
