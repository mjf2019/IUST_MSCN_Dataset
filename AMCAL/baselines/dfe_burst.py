"""DFE paper-equation reproduction with an explicit Burst input adaptation.

Wang et al., TNSE 2025, DOI 10.1109/TNSE.2025.3535577, Fig.2,
Eqs.6/7/12/13/14. The paper does not specify a tensor Gaussian estimator,
eta/lambda, all six beta values, or the voting k; see BURST_BASELINES.md.
No random layers are constructed during forward and no test extrema are fitted.
"""
from collections import deque
import math

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import torch.nn.functional as F


class FlowImages:
    """Keep every Burst feature, preserve order, and zero-pad to a square."""
    def __init__(self):
        self.scaler=MinMaxScaler(clip=True)

    def fit(self,x):
        self.width=x.shape[1];self.side=math.ceil(math.sqrt(self.width))
        self.scaler.fit(x)
        return self

    def transform(self,x):
        if x.shape[1]!=self.width:raise ValueError('DFE feature width changed.')
        values=self.scaler.transform(x).astype(np.float32)
        return np.pad(values,((0,0),(0,self.side**2-self.width))).reshape(-1,1,self.side,self.side)


class ResidualStage(nn.Module):
    def __init__(self,cin,cout,count,stride):
        super().__init__()
        layers=[]
        for i in range(count):
            layers.extend([nn.Conv2d(cin if i==0 else cout,cout,3,
                                    stride=stride if i==0 else 1,padding=1,bias=False),
                           nn.BatchNorm2d(cout),nn.ReLU()])
        self.main=nn.Sequential(*layers)
        self.shortcut=nn.Identity() if cin==cout and stride==1 else nn.Sequential(
            nn.Conv2d(cin,cout,1,stride=stride,bias=False),nn.BatchNorm2d(cout))

    def forward(self,x):return F.relu(self.main(x)+self.shortcut(x))


class DFEBackbone(nn.Module):
    """13 main 3x3 convolutions, stride-2 at 3/6/10, six residual stages."""
    def __init__(self):
        super().__init__()
        self.stages=nn.ModuleList([ResidualStage(*spec) for spec in
            [(1,64,2,1),(64,64,2,2),(64,64,1,1),
             (64,128,2,2),(128,128,2,1),(128,256,4,2)]])
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,256))

    def forward(self,x,return_features=False):
        features=[]
        for stage in self.stages:x=stage(x);features.append(x)
        z=self.fc(self.pool(x).flatten(1))
        return (z,features) if return_features else z


class GaussianCompression(nn.Module):
    """Eq.6 mixture bound using Eq.7 diagonal KL in a common output space.

For each of the six transitions, each sample's output feature map defines
one Gaussian component: channel means and spatial diagonal variances.
Pairwise KL is between samples of that SAME transition, not different-width
layers. This tensor estimator is a declared implementation choice.
"""
    def __init__(self,betas=(.01,.02,.03,.04,.05,.05),variance_floor=1e-4):
        super().__init__()
        self.register_buffer('betas',torch.tensor(betas))
        self.variance_floor=variance_floor

    @staticmethod
    def pairwise_kl(mu,var):
        # KL(P || Q); sum over dimensions, as in Eq.7 (not mean over channels).
        logdet=var.log().sum(1)
        trace=var @ var.reciprocal().T
        squared=(mu.square() @ var.reciprocal().T
                 -2*mu @ (mu/var).T+(mu.square()/var).sum(1)[None,:])
        return (.5*(logdet[None,:]-logdet[:,None]+trace+squared-mu.shape[1])).clamp_min(0)

    def forward(self,features):
        if len(features)!=len(self.betas):raise ValueError('Six feature transitions required.')
        terms=[]
        for beta,feature in zip(self.betas,features):
            flat=feature.flatten(2)
            mu=flat.mean(2);var=flat.var(2,unbiased=False).clamp_min(self.variance_floor)
            kl=self.pairwise_kl(mu,var)
            # The self-component KL is analytically zero; enforce it for roundoff.
            kl=kl.masked_fill(torch.eye(len(mu),device=mu.device,dtype=torch.bool),0)
            bound=-(torch.logsumexp(-kl,dim=1)-math.log(len(mu))).mean()
            terms.append(beta*bound)
        return torch.stack(terms).sum()


def constrained_triplet(anchor,positive,negative,alpha1=.5,alpha2=2.,eta=.5):
    dp=torch.linalg.vector_norm(anchor-positive,dim=1)
    dn=torch.linalg.vector_norm(anchor-negative,dim=1)
    # Eq.12 literally uses max(dp, alpha2). ReLU(dp-alpha2) has identical
    # gradients but omits the constant eta*alpha2; retain the printed equation.
    return F.relu(dp-dn+alpha1).mean()+eta*dp.clamp_min(alpha2).mean()


def triplet_indices(y,anchors,rng):
    positives=[];negatives=[]
    for i in anchors:
        same=np.flatnonzero((y==y[i])&(np.arange(len(y))!=i))
        other=np.flatnonzero(y!=y[i])
        if not len(same) or not len(other):raise ValueError('Triplets need two rows/class and two classes.')
        positives.append(rng.choice(same));negatives.append(rng.choice(other))
    return np.asarray(positives),np.asarray(negatives)


class TemplateLibrary:
    """Separate templates; acquired labels replace oldest same-class templates.

    FIFO replacement is a declared causal benchmark adaptation of DFE stage 3;
    it never averages embeddings into class centroids.
    """
    def __init__(self,x,y,per_class=20,seed=42):
        self.per_class=per_class;self.rows={}
        rng=np.random.default_rng(seed)
        for label in np.unique(y):
            ids=rng.choice(np.flatnonzero(y==label),min(per_class,int((y==label).sum())),replace=False)
            self.rows[int(label)]=deque([x[i].copy() for i in ids],maxlen=per_class)

    def add(self,x,y):
        self.rows.setdefault(int(y),deque(maxlen=self.per_class)).append(x.copy())

    def arrays(self):
        rows=[(row,label) for label,values in sorted(self.rows.items()) for row in values]
        return np.stack([r[0] for r in rows]),np.asarray([r[1] for r in rows])


def knn_vote(templates,labels,queries,k=1):
    predictions=[];confidence=[]
    for start in range(0,len(queries),256):
        query=queries[start:start+256]
        # Avoid [queries,templates,256/512] allocation for AF full-source support.
        distance=np.sqrt(np.maximum(0,(query*query).sum(1)[:,None]
                                     +(templates*templates).sum(1)[None,:]-2*query@templates.T))
        nearest=np.argsort(distance,axis=1,kind='stable')[:,:min(k,len(labels))]
        for row,ids in zip(distance,nearest):
            values,counts=np.unique(labels[ids],return_counts=True)
            tied=values[counts==counts.max()]
            winner=min(tied,key=lambda c:(row[ids][labels[ids]==c].mean(),int(c)))
            predictions.append(winner);confidence.append(float(counts.max()/len(ids)))
    return np.asarray(predictions),np.asarray(confidence)


def embed(model,x,device,batch=64):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(x[i:i+batch],device=device)).cpu().numpy()
                               for i in range(0,len(x),batch)])
