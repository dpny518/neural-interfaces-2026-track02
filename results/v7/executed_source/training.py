"""Fit-only neural routines; held-out arrays are never accepted here."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from runtime import MultiscaleNet


def masks(cov, rng, n_bands, n_views):
    n, _, channels, _ = cov.shape
    cm = rng.random((n, channels)) < .15
    cm[cm.all(axis=1), 0] = False
    bm = rng.random((n, n_bands)) < .2
    bm[:,6] = False
    bm = np.tile(bm, (1,n_views))
    cm = torch.as_tensor(cm, device=cov.device)
    bm = torch.as_tensor(bm, device=cov.device)
    keep = (~cm).to(cov.dtype)
    out = cov * keep[:,None,:,None] * keep[:,None,None,:] + torch.diag_embed(cm.to(cov.dtype))[:,None]
    out = torch.where(bm[:,:,None,None], torch.eye(channels, device=cov.device), out)
    entries = (bm[:,:,None] | cm[:,None,:]).flatten(1)
    if not entries.any():
        out[:,0] = torch.eye(channels, device=cov.device)
        entries[:,:channels] = True
    return out, entries


def pretrain(net, X, seed, config):
    target = X.diagonal(dim1=-2,dim2=-1).clamp_min(1e-6).log().flatten(1)
    center = target.mean(0)
    scale = target.std(0,correction=0).clamp_min(.1)
    target = (target-center)/scale
    with torch.random.fork_rng(devices=[X.device.index] if X.is_cuda else []):
        torch.manual_seed(seed+100000)
        decoder = nn.Sequential(nn.Linear(net.normalizer.num_features,128),nn.GELU(),
                                nn.Linear(128,target.shape[1])).to(X.device)
    params = [net.spatial_weight]+list(net.normalizer.parameters())+list(decoder.parameters())
    opt = torch.optim.AdamW(params,lr=.001,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30,eta_min=.0001)
    rng = np.random.default_rng(seed+100001)
    history = []
    for epoch in range(30):
        losses = []; net.train(); decoder.train()
        order = rng.permutation(len(X))
        for start in range(0,len(X),96):
            idx = torch.as_tensor(order[start:start+96],device=X.device)
            if len(idx)<2:continue
            masked, entries = masks(X[idx],rng,config['n_bands'],config['n_views'])
            pred = decoder(net.normalizer(net.features(masked)))
            loss = (F.smooth_l1_loss(pred,target[idx],reduction='none')*entries).sum()/entries.sum()
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(params,5.);opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
    return net.spatial_weight.detach().clone(), {
        'history':history,'target_center':center.cpu().tolist(),'target_scale':scale.cpu().tolist(),
        'n_fitting_windows':len(X),'labels_used':False,'retained_parameters':['spatial_weight']}


def fit_neural(cov, labels, weights, subjects, excluded, config, recipe, seed, epochs, device):
    assert not set(subjects)&set(excluded)
    assert len(cov)==len(labels)==len(weights)==len(subjects)
    assert len(cov)>1 and np.isfinite(cov).all()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    net = MultiscaleNet(**config).to(device)
    X = torch.as_tensor(cov,device=device,dtype=torch.float32)
    Y = torch.as_tensor(labels,device=device,dtype=torch.long)
    W = torch.as_tensor(weights,device=device,dtype=torch.float32)
    ssl = None
    if recipe=='masked_ssl':
        spatial, ssl = pretrain(net,X,seed,config)
        torch.manual_seed(seed)
        net = MultiscaleNet(**config).to(device)
        with torch.no_grad():net.spatial_weight.copy_(spatial)
    opt = torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=120,eta_min=.0001)
    rng = np.random.default_rng(seed)
    aug = np.random.default_rng(seed+300000)
    history = []
    for epoch in range(epochs):
        net.train(); order=rng.permutation(len(X)); losses=[]
        for start in range(0,len(X),96):
            ids = order[start:start+96]
            if len(ids)<2:continue
            idx = torch.as_tensor(ids,device=device)
            batch = X[idx]
            gains = torch.exp(torch.randn(len(ids),1,41,device=device)*.08)
            batch = batch*gains.unsqueeze(-1)*gains.unsqueeze(-2)
            if recipe=='cross_subject_mixup':
                partners=np.arange(len(ids)); lam=np.ones(len(ids),dtype=np.float32)
                for j in range(len(ids)):
                    eligible=np.flatnonzero(subjects[ids]!=subjects[ids[j]])
                    if len(eligible) and aug.random()<.5:
                        partners[j]=aug.choice(eligible);lam[j]=aug.beta(.2,.2)
                p=torch.as_tensor(partners,device=device)
                t=torch.as_tensor(lam,device=device)
                batch=batch*t[:,None,None,None]+batch[p]*(1-t)[:,None,None,None]
                logits=net(batch)
                loss=(t*W[idx]*F.cross_entropy(logits,Y[idx],reduction='none',label_smoothing=.05)+
                      (1-t)*W[idx][p]*F.cross_entropy(logits,Y[idx][p],reduction='none',label_smoothing=.05)).mean()
            else:
                loss=(W[idx]*F.cross_entropy(net(batch),Y[idx],reduction='none',label_smoothing=.05)).mean()
            w=F.normalize(net.spatial_weight,dim=-1)
            gram=w@w.transpose(-1,-2)
            loss=loss+.001*((gram-torch.eye(gram.shape[-1],device=device))**2).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses))})
    net.eval()
    return net.cpu(), {'training_history':history,'ssl':ssl,'seed':seed,'fixed_epochs':epochs,
                       'fitting_subjects':sorted(set(subjects)),'excluded_subjects':sorted(excluded)}
