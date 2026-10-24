import torch,numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
LABELS=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();ys=[];ps=[];loss=0;n=0;ce=torch.nn.CrossEntropyLoss(reduction='sum')
    for x,y in loader:
        x,y=x.to(device),y.to(device);z=model(x);loss+=ce(z,y).item();n+=len(y);ys.extend(y.cpu().tolist());ps.extend(z.argmax(1).cpu().tolist())
    pr,re,f,_=precision_recall_fscore_support(ys,ps,labels=list(range(10)),zero_division=0)
    return {'test_loss':loss/n,'test_accuracy':float(np.mean(np.array(ys)==np.array(ps))),'macro_precision':float(pr.mean()),'macro_recall':float(re.mean()),'macro_f1':float(f.mean()),'per_class_recall':re.tolist(),'confusion_matrix':confusion_matrix(ys,ps,labels=list(range(10))).tolist()}
