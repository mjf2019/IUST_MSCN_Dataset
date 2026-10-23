"""RAL-CD reconstruction of Wassermann et al., TNSM 2021, section V-E.

Query/reward equations are inherited from the existing RAL implementation.
Only committee-query rewards enter ADWIN; epsilon-only labels do not.
"""
from collections import deque
from importlib.metadata import version
import numpy as np
from .ral_existing import RAL, TrainingData


class RewardADWIN:
    def __init__(self, backend='river', delta=.002, clock=32):
        self.backend = backend
        if backend == 'river':
            from river.drift import ADWIN
            self.detector = ADWIN(delta=delta, clock=clock)
            # River's public update resets W1 on the NEXT event after drift.
            # Keep the ADWIN2 retained window, as in scikit-multiflow ADWIN.
            if not callable(getattr(getattr(self.detector, '_helper', None), 'update', None)):
                raise RuntimeError('Unsupported River ADWIN interface; use skmultiflow backend.')
            self.library_version = version('river')
        else:
            from skmultiflow.drift_detection import ADWIN
            self.detector = ADWIN(delta=delta)
            self.detector.set_clock(clock)
            self.library_version = version('scikit-multiflow')

    def update(self, reward):
        if self.backend == 'river':
            changed = bool(self.detector._helper.update(float(reward)))
        else:
            self.detector.add_element(float(reward))
            changed = bool(self.detector.detected_change())
        return changed, int(self.detector.width)


class RALCD(RAL):
    def __init__(self, *args, adwin_backend='river', adwin_delta=.002,
                 adwin_clock=32, **kwargs):
        super().__init__(*args, **kwargs)
        if self._model_update_frequency != 1:
            raise ValueError('RAL-CD runner requires one refit per acquired label.')
        self.monitor = RewardADWIN(adwin_backend, adwin_delta, adwin_clock)
        self.reward_events = deque()
        self.feedback = deque()
        self.drifts = []
        self.initial_memory_removed = False
        self.refits = 0
        self.max_fit_rows = 0
        self.last_reward = None
        self.last_drift = False

    def acquire_label(self, x, y, committee_decision, decisions):
        # All prediction/reward computations happen before adaptation.
        self.last_reward = None
        self.last_drift = False
        paid_index = self._n_acquired_samples + 1
        changed = False
        if committee_decision:
            yp = self._committee_classifier.predict(np.asarray(x).reshape(1, -1))[0]
            self.last_reward = float(self.reward(y, yp))
            self._update_alpha(self.last_reward, decisions, committee_decision)
            self._update_uncertainty_threshold(self.last_reward)
            self.reward_events.append(paid_index)
            changed, width = self.monitor.update(self.last_reward)
            if not 1 <= width <= len(self.reward_events):
                raise RuntimeError('ADWIN width cannot be mapped to reward-event history.')
            while len(self.reward_events) > width:
                self.reward_events.popleft()
        self._n_acquired_samples = paid_index
        self.feedback.append((paid_index, np.asarray(x).copy(), y))
        self._training_data_samples.append(np.asarray(x).copy(), y)
        if changed:
            cutoff = self.reward_events[0]
            # Keep epsilon-only acquired labels inside the retained time interval.
            while self.feedback and self.feedback[0][0] < cutoff:
                self.feedback.popleft()
            before = len(self._training_data_samples.y)
            self._training_data_samples = TrainingData(
                [row[1] for row in self.feedback], [row[2] for row in self.feedback])
            self.initial_memory_removed = True
            self.last_drift = True
            self.drifts.append(dict(seen=self._n_seen, labels=paid_index,
                reward=self.last_reward, reward_window=int(self.monitor.detector.width),
                first_retained_label=cutoff, before_rows=before,
                retained_rows=len(self._training_data_samples.y)))
        # Auxiliary feedback history only needs the earliest retained reward onward.
        if self.reward_events:
            while self.feedback and self.feedback[0][0] < self.reward_events[0]:
                self.feedback.popleft()
        X, labels = self._training_data_samples.X, self._training_data_samples.y
        if len(labels) < 5:
            raise RuntimeError('ADWIN retained fewer than kNN k=5 training rows.')
        for learner in self._committee_classifiers:
            learner.fit(X, labels)
        self._committee_classifier.fit(X, labels)
        self.refits += 1
        self.max_fit_rows = max(self.max_fit_rows, len(labels))

    def statistics(self):
        return dict(drift_count=len(self.drifts), drifts=self.drifts,
            retained_training_rows=len(self._training_data_samples.y),
            initial_memory_removed=self.initial_memory_removed, refits=self.refits,
            max_fit_rows=self.max_fit_rows, reward_window=int(self.monitor.detector.width),
            adwin_backend=self.monitor.backend, adwin_library_version=self.monitor.library_version)
