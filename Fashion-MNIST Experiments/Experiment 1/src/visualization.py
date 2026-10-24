from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_history(rows,path,title):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(1,2,figsize=(10,4))
    for k,l in [('train_loss','Train'),('val_loss','Validation')]:ax[0].plot([r[k] for r in rows],label=l)
    for k,l in [('train_accuracy','Train'),('val_accuracy','Validation')]:ax[1].plot([r[k] for r in rows],label=l)
    ax[0].set(xlabel='Epoch',ylabel='Loss');ax[1].set(xlabel='Epoch',ylabel='Accuracy');[a.legend() for a in ax];fig.suptitle(title);fig.tight_layout();fig.savefig(p,dpi=200);fig.savefig(p.with_suffix('.pdf'));plt.close(fig)
