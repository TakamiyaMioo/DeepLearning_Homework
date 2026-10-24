from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
def split_indices(n=60000,train_size=50000,seed=42):
    ix=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).tolist(); tr,va=ix[:train_size],ix[train_size:]
    assert len(tr)+len(va)==n and not set(tr)&set(va) and min(ix)>=0 and max(ix)<n
    return tr,va
def get_loaders(root='data',batch_size=128,split_seed=42,download=False,num_workers=2):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);tf=transforms.ToTensor()
    try:
        ds=datasets.FashionMNIST(str(root),train=True,download=download,transform=tf); test=datasets.FashionMNIST(str(root),train=False,download=download,transform=tf)
    except RuntimeError as e: raise RuntimeError('Fashion-MNIST 缺失；服务器首次运行加 --download') from e
    f=root/'splits'/f'split_seed_{split_seed}.pt';f.parent.mkdir(parents=True,exist_ok=True)
    if f.exists():
        d=torch.load(f,map_location='cpu',weights_only=False);tr,va=d['train_indices'],d['val_indices']
    else:
        tr,va=split_indices(len(ds),50000,split_seed);torch.save({'train_indices':tr,'val_indices':va},f)
    if len(tr)!=50000 or len(va)!=10000 or set(tr)&set(va): raise ValueError('split invalid')
    return (DataLoader(Subset(ds,tr),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(split_seed),num_workers=num_workers),DataLoader(Subset(ds,va),batch_size=batch_size,shuffle=False,num_workers=num_workers),DataLoader(test,batch_size=batch_size,shuffle=False,num_workers=num_workers))
