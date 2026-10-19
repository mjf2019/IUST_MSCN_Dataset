"""Paper-based CALMID (2021) and MicFoal (2023), with paid-only stream APIs.

Equations/algorithm line mappings and MOA/River differences are documented in
CALMID_MICFOAL.md. Prediction and label requests accept no current true label.
An unqueried row advances windows using placeholders only. Native random-query
rules remain intact; the caller supplies an additional strict global cap.
"""
from collections import Counter, deque
from dataclasses import dataclass
import math
import numpy as np

VERSION = 1


def features(x):
    return {i:float(v) for i,v in enumerate(x)}


def normalized(votes, classes):
    result={int(c):max(0.,float(votes.get(c,0.))) for c in classes}
    total=sum(result.values())
    return {c:v/total for c,v in result.items()} if total else {c:1/len(classes) for c in classes}


class LabelWindows:
    """Time-aligned Wt/Wp/Wr; random counts exclude active-query labels."""
    def __init__(self, size):
        self.size=size
        self.rows=deque()
        self.true=Counter();self.pred=Counter();self.random=Counter();self.tp=Counter()

    def _count(self, row, delta):
        y,p,r=row
        for counts,key in ((self.true,y),(self.pred,p),(self.random,r)):
            if key is not None:
                counts[key]+=delta
                if counts[key]==0:del counts[key]
        if y is not None and y==p:
            self.tp[y]+=delta
            if self.tp[y]==0:del self.tp[y]

    def append(self, y=None, prediction=None, random_selected=False):
        if y is None and prediction is not None:
            raise ValueError('An unqueried row must contain only placeholders.')
        row=(y,prediction,y if random_selected else None)
        if len(self.rows)==self.size:self._count(self.rows.popleft(),-1)
        self.rows.append(row);self._count(row,1)

    def macro_f1(self):
        # Paper Sec.3.5: classes present in Wt, not the global class vocabulary.
        if not self.true:return None
        return sum(2*self.tp[c]/(self.true[c]+self.pred[c]) for c in self.true)/len(self.true)


@dataclass(frozen=True)
class Decision:
    step:int
    prediction:int
    second:int
    confidence:float
    margin:float
    random:bool=False
    uncertain:bool=False
    selective:bool=False


class ActiveStream:
    def begin_stream(self, seed, budget):
        if not 0<=budget<=1:raise ValueError('Budget must be in [0,1].')
        self.seed=seed;self.rng=np.random.default_rng(seed)
        self.initial_budget=self.budget=float(budget)
        self.seen=self.queried=0;self.pending=None
        self.query_counts=Counter();self.last_weight=None
        self.periods=0;self.last_period_f1=None;self.drifts=0

    def predict(self, x):
        return np.asarray([self.predict_one(row) for row in x],dtype=np.int64)

    def predict_one(self, x):
        votes=self.probabilities(x)
        return max(votes,key=lambda c:(votes[c],-c))

    def _rank(self, x):
        votes=self.probabilities(x)
        order=sorted(votes,key=lambda c:(-votes[c],c))
        first=order[0];second=order[1] if len(order)>1 else first
        return first,second,votes[first],votes[first]-votes[second] if len(order)>1 else votes[first]

    def _start_request(self):
        if self.pending is not None:raise RuntimeError('Pay/complete the pending query before the next request.')
        self.seen+=1;self.clock+=1

    def _finish_request(self, decision, selected):
        if selected:self.pending=decision
        else:
            self.windows.append()
            self._periodic()
        return bool(selected),decision

    def _paid(self, decision):
        if decision is not self.pending or decision.step!=self.seen:
            raise RuntimeError('Feedback must match the current paid request.')
        self.pending=None;self.queried+=1
        for key in ('random','uncertain','selective'):
            if getattr(decision,key):self.query_counts[key]+=1

    def _periodic(self):pass

    def statistics(self):
        return dict(implementation_version=VERSION,native_budget_final=self.budget,
            native_budget_initial=self.initial_budget,stream_seen=self.seen,
            acquired_labels=self.queried,query_counts=dict(self.query_counts),
            last_training_weight=self.last_weight,periodic_evaluations=self.periods,
            last_period_macro_f1=self.last_period_f1,drifts=self.drifts,
            random_window_labels=sum(self.windows.random.values()),
            current_window_classes=len(self.windows.true))


class CALMID(ActiveStream):
    """Paper Algorithms 1--3; fixed class vocabulary and Poisson(weight)."""
    def __init__(self, classes, seed=42, models=10, window=500, threshold=.5,
                 random_ratio=.1, step=.01, budget=.2, adwin_delta=.002):
        from river import drift, tree
        self.classes=tuple(sorted(int(c) for c in classes))
        if len(self.classes)<2 or min(models,window)<1:raise ValueError('CALMID needs >=2 classes and positive model/window sizes.')
        if not 0<random_ratio<=1 or not 0<threshold<=1 or not 0<step<1:raise ValueError('Invalid CALMID thresholds.')
        self.models=models;self.window=window;self.theta=threshold
        self.random_ratio=random_ratio;self.step=step;self.adwin_delta=adwin_delta
        self.windows=LabelWindows(window)
        self.thresholds={(i,j):threshold for i in self.classes for j in self.classes}
        self.cache_size=max(1,math.ceil(window*random_ratio/len(self.classes)))
        self.cache={c:deque(maxlen=self.cache_size) for c in self.classes}
        self.trees=[tree.HoeffdingTreeClassifier() for _ in range(models)]
        self.detectors=[drift.ADWIN(delta=adwin_delta) for _ in range(models)]
        self.clock=0;self.begin_stream(seed,budget)

    def probabilities(self, x):
        x=features(x);total=Counter()
        for tree in self.trees:
            votes=tree.predict_proba_one(x);mass=sum(votes.values())
            if mass>0:
                for c,v in votes.items():total[c]+=v/mass
        return normalized(total,self.classes)

    def imbalance(self, y):
        count=sum(self.windows.random.values())
        # Eq.3 otherwise has 0/0 or division by zero. The author's count floor
        # is retained; an entirely empty random window has neutral imbalance.
        return max(1,self.windows.random[y])*len(self.classes)/count if count else 1.

    @staticmethod
    def difficulty(decision, y):
        if y==decision.prediction:return 1-decision.margin
        return (1+decision.margin)*math.exp(1 if y==decision.second else 2)

    def request(self, x, can_query=True):
        self._start_request();first,second,confidence,margin=self._rank(x)
        random_selected=bool(can_query and self.rng.random()<self.random_ratio)
        uncertain=selective=False
        # Algorithm 1: random first; budget gates uncertainty/selective only.
        if can_query and not random_selected and self.queried/self.seen<self.budget:
            threshold=self.thresholds[first,second]
            uncertain=margin<=threshold
            if not uncertain:
                remaining=self.budget-self.queried/self.seen
                selective=self.rng.random()<remaining/(remaining+margin-threshold)
        d=Decision(self.seen,first,second,confidence,margin,random_selected,bool(uncertain),bool(selective))
        return self._finish_request(d,random_selected or uncertain or selective)

    def _threshold_update(self, d, y):
        key=d.prediction,d.second
        if d.uncertain:
            imb=self.imbalance(y)
            decreases=(1+int(imb>.5)) if y==d.prediction else int(y==d.second and imb>.5)
            self.thresholds[key]*=(1-self.step)**decreases
        elif d.selective and y==d.second:
            self.thresholds[key]=max(self.theta,self.thresholds[key]*(1+self.step))

    def _fit_weight(self, tree, x, y, weight):
        # Algorithm 3 line 10: sample a Poisson count with the computed weight.
        k=int(self.rng.poisson(weight))
        if k:tree.learn_one(features(x),y,w=k)

    def _replace_worst(self):
        from river import drift, tree
        worst=max(range(self.models),key=lambda i:self.detectors[i].estimation)
        fresh=tree.HoeffdingTreeClassifier()
        samples=sorted((row for rows in self.cache.values() for row in rows),key=lambda row:row[0])
        for arrived,x,y,diff in samples:
            base=1. if diff is None else math.log1p(diff+1/self.imbalance(y))
            weight=math.exp((arrived-self.clock)/self.window)*base
            self._fit_weight(fresh,x,y,weight)
        self.trees[worst]=fresh;self.detectors[worst]=drift.ADWIN(delta=self.adwin_delta)
        self.drifts+=1
        return worst

    def _detect_after_learning(self, x, y):
        changed=False
        for tree,detector in zip(self.trees,self.detectors):
            before=detector.estimation
            # Paper Alg.1 checks drift after learning; author code uses this
            # post-update per-member error as well, never an unqueried label.
            detector.update(int(tree.predict_one(features(x))!=y))
            changed|=bool(detector.drift_detected and detector.estimation>before)
        if changed:self._replace_worst()

    def update(self, x, y, decision):
        y=int(y)
        if y not in self.classes:raise ValueError('CALMID has a fixed class vocabulary.')
        self._paid(decision)
        # Algorithm 2 executes before Algorithm 1 pushes the Null placeholder.
        self._threshold_update(decision,y)
        self.windows.append(y,decision.prediction,decision.random)
        diff=self.difficulty(decision,y)
        weight=math.log1p(diff+1/self.imbalance(y))  # Eq.2, no added unit weight.
        self.last_weight=weight
        for tree in self.trees:self._fit_weight(tree,x,y,weight)
        self.cache[y].append((self.clock,np.asarray(x).copy(),y,diff))
        self._detect_after_learning(x,y)

    def fit_source(self, x, y):
        # Benchmark initialization: ALL available TRAIN labels, excluded from
        # target budget. This replaces the paper's first-z bootstrap, not AL.
        for row,label in zip(x,y):
            label=int(label);self.clock+=1;prediction=self.predict_one(row)
            self.windows.append(label,prediction,True)
            for tree in self.trees:
                k=int(self.rng.poisson(6))
                if k:tree.learn_one(features(row),label,w=k)
            self.cache[label].append((self.clock,np.asarray(row).copy(),label,None))
            self._detect_after_learning(row,label)
        return self

    def statistics(self):
        result=super().statistics()
        result.update(threshold_matrix=[[self.thresholds[i,j] for j in self.classes] for i in self.classes],
            per_class_cache_size=self.cache_size,cached_samples=sum(map(len,self.cache.values())),
            bootstrap_protocol='available TRAIN labels; Poisson(6); neutral source-cache weights',
            bagging_protocol='paper Poisson(computed weight), including age-decayed replacement')
        return result


class MicFoal(ActiveStream):
    """Paper Algorithms 1--3, Eq.1--3 and periodic paid-history evaluation."""
    def __init__(self, seed=42, models=10, window=500, threshold=.9,
                 random_ratio=.1, step=.1, budget=.2, adaptive_budget=True,
                 low_f1=.6, high_f1=.9):
        from .weighted_arf import WeightedARFClassifier
        if min(models,window)<1 or not 0<random_ratio<=1 or not 0<threshold<=1 or not 0<step<1:
            raise ValueError('Invalid MicFoal model/window/threshold parameters.')
        if not 0<=low_f1<high_f1<=1:raise ValueError('Invalid MicFoal F1 bounds.')
        self.window=window;self.theta=threshold;self.random_ratio=random_ratio;self.step=step
        self.adaptive_budget=adaptive_budget;self.low_f1=low_f1;self.high_f1=high_f1
        self.model=WeightedARFClassifier(n_models=models,lambda_value=6,seed=seed)
        self.thresholds={};self.windows=LabelWindows(window);self.clock=0
        self.begin_stream(seed,budget)

    def begin_stream(self, seed, budget):
        super().begin_stream(seed,budget)
        self.backend_drifts_at_start=self.model.n_drifts_detected()

    def _register(self, y):
        if y not in self.thresholds:self.thresholds[y]=self.theta

    def probabilities(self, x):
        classes=tuple(sorted(self.thresholds)) or (0,)
        return normalized(self.model.predict_proba_one(features(x)),classes)

    def request(self, x, can_query=True):
        self._start_request();first,second,confidence,margin=self._rank(x)
        threshold=self.thresholds.get(first,self.theta)
        uncertain=selective=False
        # Algorithm 2 draws random AFTER uncertain/selective; random is outside
        # the native ratio gate. The external can_query still imposes the cap.
        if can_query and self.queried/self.seen<=self.budget:
            uncertain=confidence<=threshold
            if not uncertain:
                remaining=self.budget-self.queried/self.seen
                selective=self.rng.random()<remaining/(remaining+confidence-threshold)
        random_selected=bool(can_query and self.rng.random()<self.random_ratio)
        d=Decision(self.seen,first,second,confidence,margin,random_selected,bool(uncertain),bool(selective))
        return self._finish_request(d,random_selected or uncertain or selective)

    def training_weight(self, y, decision):
        nclasses=len(self.windows.true);nall=sum(self.windows.random.values());ny=self.windows.random[y]
        difficulty=1+self.thresholds.get(y,self.theta)-decision.confidence if y==decision.prediction else 2*(1+nclasses)
        if not ny:return 1+math.log1p(difficulty+nall),0.
        mu=ny*nclasses/nall
        return 1+math.log1p(difficulty+1/mu),mu

    def update(self, x, y, decision):
        y=int(y);self._paid(decision)
        self.windows.append(y,decision.prediction,decision.random)
        weight,mu=self.training_weight(y,decision)
        self.last_weight=weight
        self.model.learn_one(features(x),y,w=weight)
        if y==decision.prediction and decision.uncertain:
            self.thresholds[decision.prediction]=self.thresholds.get(decision.prediction,self.theta)*(1-self.step)**(2 if mu>=1 else 1)
        elif y!=decision.prediction and not decision.uncertain:
            # Eq.3 has neither the Java max(theta,...) floor nor a unit cap.
            self.thresholds[decision.prediction]=self.thresholds.get(decision.prediction,self.theta)*(1+self.step)
        self._register(y)  # expand only AFTER receiving an unseen paid label.
        self._periodic()

    def _periodic(self):
        if not self.seen or self.seen%self.window:return
        self.periods+=1;score=self.windows.macro_f1();self.last_period_f1=score
        if score is None or not self.adaptive_budget or not self.initial_budget:return
        # Sec.3.5 specifies direction, not magnitudes. These rates and bounds
        # are taken from the official Java adjustBudget/PeriodicEvaluate.
        if score<self.low_f1:self.budget*=1+(self.low_f1-score)/10
        elif score>self.high_f1:self.budget*=1-(score-self.high_f1)/100
        self.budget=min(min(1.,2*self.initial_budget),max(min(self.random_ratio,self.initial_budget),self.budget))

    def fit_source(self, x, y):
        for row,label in zip(x,y):
            label=int(label);prediction=self.predict_one(row);self.clock+=1
            self._register(label);self.windows.append(label,prediction,True)
            self.model.learn_one(features(row),label,w=1.)
        return self

    def statistics(self):
        result=super().statistics()
        result.update(threshold_vector=dict(self.thresholds),adaptive_budget=self.adaptive_budget,
            drifts=self.model.n_drifts_detected()-self.backend_drifts_at_start,
            backend_drifts=self.model.n_drifts_detected(),
            bagging_protocol='sample weight times one ARF Poisson(6) draw; foreground and background',
            bootstrap_protocol='all available TRAIN labels; native ARF initialization')
        return result
