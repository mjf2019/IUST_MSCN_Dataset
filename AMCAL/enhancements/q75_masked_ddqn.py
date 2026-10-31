"""Two-action DDQN with one fixed TRAIN-q75 action mask in behavior and bootstrap."""
import math
import torch
from .two_action_ddqn import QueryDDQN


class Q75MaskedDDQN(QueryDDQN):
    def __init__(self,spec,actor_state,device,seed=42,*,threshold,classes,**kwargs):
        if not isinstance(classes,int) or classes<2 or len(spec['mean'])!=2*classes+18:
            raise ValueError('q75 class/state specification mismatch.')
        if not math.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('Invalid fixed TRAIN q75.')
        super().__init__(spec,actor_state,device,seed,**kwargs)
        self.threshold=float(threshold); self.classes=classes; self.decision_log=None

    def entropy(self,state):
        value=self._state(state)
        entropy=float(value[2*self.classes+5])
        if not 0<=entropy<=1.000001: raise ValueError('Invalid normalized Context entropy.')
        return entropy

    def eligible(self,state): return self.entropy(state)>=self.threshold

    def choose(self,state,can_query,epsilon,rng):
        entropy=self.entropy(state)
        legal=bool(can_query) and entropy>=self.threshold
        action,q=super().choose(state,legal,epsilon,rng)
        if self.decision_log is not None:
            self.decision_log.append(dict(guide_entropy_state=entropy,guide_threshold=self.threshold,
                guide_uncertainty_eligible=entropy>=self.threshold,guide_budget_eligible=bool(can_query),
                guide_action=int(action),guide_mode='AlignedQ75DDQN'))
        return action,q

    def remember(self,state,action,reward,successor,terminal,next_can_query):
        if terminal and next_can_query: raise ValueError('Terminal transition cannot permit a successor query.')
        if action==1 and not self.eligible(state): raise ValueError('TRAIN query violated the q75 action mask.')
        # The parent learner applies this mask to online argmax before gathering target Q.
        masked_next=bool(not terminal and next_can_query and self.eligible(successor))
        return super().remember(state,action,reward,successor,terminal,masked_next)

    def snapshot(self):
        saved=super().snapshot()
        saved['q75_mask']=dict(version=1,classes=self.classes,threshold=self.threshold)
        return saved

    def restore(self,saved):
        expected=dict(version=1,classes=self.classes,threshold=self.threshold)
        if saved.get('q75_mask')!=expected: raise ValueError('Resumed q75 action mask differs.')
        super().restore(saved)
        for state,action,_,successor,terminal,next_can_query in self.memory:
            if action==1 and not self.eligible(state): raise ValueError('Replay contains an illegal q75 query.')
            if next_can_query and (terminal or not self.eligible(successor)):
                raise ValueError('Replay contains an illegal q75 bootstrap action.')

