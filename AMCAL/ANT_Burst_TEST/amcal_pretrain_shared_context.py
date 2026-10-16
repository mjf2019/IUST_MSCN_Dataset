"""Supervised shared Context pretraining with a frozen CNN and fixed validation.
Writes a separate source-format run for subsequent three-action training.
No test data, online validation updates, RL or flow/class routing.
"""
from pathlib import Path
import argparse
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import amcal_burst_three_action as core


def evaluate(x,y,cnn,context,batch_size):
    context.eval()
    predictions=[]
    losses=[]
    with torch.no_grad():
        for start in range(0,len(y),batch_size):
            target=y[start:start+batch_size]
            logits=cnn(context(x[start:start+batch_size]))
            predictions.extend(logits.argmax(1).cpu().tolist())
            losses.append(float(nn.functional.cross_entropy(logits,target,reduction='sum').item()))
    truth=y.cpu().tolist()
    return dict(accuracy=float(accuracy_score(truth,predictions)),
                macro_f1=float(f1_score(truth,predictions,average='macro',zero_division=0)),
                weighted_f1=float(f1_score(truth,predictions,average='weighted',zero_division=0)),
                loss=sum(losses)/len(y))


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-output',type=Path,required=True,help='Existing source run containing the frozen CNN.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=root/'Dataset')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--init',choices=['fresh','source'],default='fresh')
    args=p.parse_args()
    if min(args.epochs,args.patience)<1 or args.batch_size<2 or not np.isfinite(args.lr) or args.lr<=0:
        p.error('Invalid epochs, patience, learning rate or batch size.')
    if args.output.resolve()==args.source_output.resolve() or (args.output.exists() and any(args.output.iterdir())):
        p.error('Choose a fresh output directory.')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_ck=torch.load(args.source_output/'Models'/'initial_cnn_model.pth',map_location='cpu',weights_only=False)
    source_ck=torch.load(args.source_output/'Models'/'adversarial_best_context.pth',map_location='cpu',weights_only=False)
    cnn,context=core.build_models(cnn_ck,source_ck,device)
    if args.init=='fresh':
        context=core.legacy.ContextAwareNetwork(input_size=cnn_ck['input_length'],num_layers=4).to(device)
    train=pd.read_csv(args.data_dir/'oversampled_train_dataset.csv')
    val=pd.read_csv(args.data_dir/'oversampled_validation_dataset.csv')
    if len(train)<2 or len(val)==0:
        p.error('At least two training samples and one validation sample are required.')
    scaler=StandardScaler().fit(train.iloc[:,:-1])
    encoder=LabelEncoder().fit(train.iloc[:,-1])
    columns=list(train.columns)
    if train.shape[1]-1 != cnn_ck['input_length'] or len(encoder.classes_) != cnn_ck['num_classes']:
        p.error('Dataset and CNN dimensions differ.')
    tx,ty=core.tensors(train,scaler,encoder,columns,device)
    vx,vy=core.tensors(val,scaler,encoder,columns,device)
    ty=torch.tensor(ty,device=device,dtype=torch.long)
    vy=torch.tensor(vy,device=device,dtype=torch.long)
    initial_cnn={k:v.detach().cpu().clone() for k,v in cnn.state_dict().items()}
    optimizer=torch.optim.Adam(context.parameters(),lr=args.lr,weight_decay=1e-4)
    frozen=evaluate(vx,vy,cnn,nn.Identity(),args.batch_size)
    print("Frozen CNN fixed validation:",json.dumps(frozen))
    initial=evaluate(vx,vy,cnn,context,args.batch_size)
    print('Initial fixed validation:',json.dumps(initial))
    models=args.output/'Models';models.mkdir(parents=True)
    torch.save(cnn_ck,models/'initial_cnn_model.pth')
    best,stale,history=-1.,0,[]
    best_epoch=None
    for epoch in range(args.epochs):
        context.train();cnn.eval()
        # Merge a trailing singleton into its preceding batch for BatchNorm.
        batches=list(torch.randperm(len(ty),device=device).split(args.batch_size))
        if len(batches)>1 and len(batches[-1])==1:
            batches[-2]=torch.cat((batches[-2],batches[-1]));batches.pop()
        loss_sum=0.
        for indices in batches:
            optimizer.zero_grad()
            logits=cnn(context(tx[indices]))
            loss=nn.functional.cross_entropy(logits,ty[indices])
            loss.backward();optimizer.step()
            loss_sum+=float(loss.item())*len(indices)
        validation=evaluate(vx,vy,cnn,context,args.batch_size)
        best,stale,improved=core.early_stopping_step(validation['accuracy'],best,stale)
        row=dict(epoch=epoch+1,train_loss=loss_sum/len(ty),validation=validation,
                 epochs_without_improvement=stale)
        history.append(row);print(json.dumps(row))
        if improved:
            best_epoch=epoch+1
            # source-compatible format; 'aligned' is the downstream Context
            # objective metadata. Pretraining itself always uses plain CE.
            checkpoint=copy.deepcopy(source_ck)
            checkpoint.update(context_model_state_dict={k:v.detach().cpu().clone() for k,v in context.state_dict().items()},
                              context_loss_mode='aligned',pretraining_objective='cross_entropy',
                              pretraining_init=args.init,best_epoch=best_epoch,validation=validation,
                              validation_protocol='fixed held-out validation; no updates',training_args=vars(args))
            torch.save(checkpoint,models/'adversarial_best_context.pth')
            torch.save(dict(cnn_checkpoint=cnn_ck,context_checkpoint=checkpoint,
                            scaler=scaler,encoder=encoder,columns=columns),models/'context_pretrain_best.pth')
        (args.output/'training_history.json').write_text(json.dumps(history,indent=2))
        if stale>=args.patience:
            print('Early stopping at epoch',epoch+1);break
    assert all(torch.equal(initial_cnn[k],v.detach().cpu()) for k,v in cnn.state_dict().items())
    (args.output/'manifest.json').write_text(json.dumps(dict(args=vars(args),frozen_validation=frozen,initial_validation=initial,
        best_epoch=best_epoch,best_validation_accuracy=best,completed_epochs=len(history),
        classifier_unchanged=True,pretraining_objective='cross_entropy',
        context_training_mode='train with BatchNorm/dropout; fixed eval validation'),default=str,indent=2))
    print('Best fixed validation:',best,'epoch:',best_epoch)
    print('Source run for subsequent three-action training:',args.output)


if __name__=='__main__':
    main()
