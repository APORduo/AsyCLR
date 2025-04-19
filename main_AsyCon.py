
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from utils import misc
from utils.config import cfg,unknow
from utils.util import set_logger,AverageMeter,TwoCropTransform
from models import model_dict
cfg.trainer = 'AsyCon'
cfg.epoch = 200
cfg.lr = 0.1 
cfg.momentum = 0.9
cfg.weight_decay = 5e-4
cfg.test_freq = 20
cfg.img_size = {"mini_imagenet": 84, "cifar100": 32,'cub200':224}[cfg.dataset]
cfg.lamb = 1.0
cfg.lr_min = 6e-5
    
cfg.prefix = ''
cfg.tmp = 0.5
cfg.merge_from_list(unknow)
logname =  f"{cfg.trainer}_{cfg.encoder}_{cfg.epoch}.log" 

set_logger(cfg,logname=logname,prefix=cfg.prefix)
args = cfg

if cfg.seed >= 0:
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

class AsyCon(nn.Module):
    def __init__(self, model, num_ftrs):
        super().__init__()
        self.backbone = model
        self.projection_head = SimSiamProjectionHead(
            input_dim=num_ftrs,
            hidden_dim=512,
            output_dim=128
        )
        self.prediction_head = SimSiamPredictionHead(
            input_dim=128,
            hidden_dim=64,
            output_dim=128
        )

    def forward(self, x):
        x = self.backbone(x)
        z = self.projection_head(x)
        p = self.prediction_head(z)
        #z = z.detach()
        if not self.training:
           return p
        return z, p
    

def adjust_learning_rate(optimizer, init_lr, epoch, args,lr_min = 5e-4):
    """Decay the learning rate based on schedule"""
    #cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epoch))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = param_group['init_lr']
        else:
            lr = param_group['init_lr'] * 0.5 * (1. + math.cos(math.pi * epoch / args.epoch))
            
            param_group['lr'] = np.clip(lr,lr_min,0.1)
            
backbone = model_dict[args.encoder]()
backbone.fc = nn.Identity()
feat_dim = backbone.feat_dim
model = AsyCon(backbone, feat_dim)
model.cuda()


def main():
   
    trainset, trainloader, testloader =misc.get_dataloader(args, session = 0)
    trainset.transform = TwoCropTransform(trainset.transform) 

    params =[
                {'params':model.backbone.parameters(), 'init_lr':args.lr,'fix_lr': False},
                {'params':model.projection_head.parameters(),'init_lr':args.lr, 'fix_lr': False},
                {'params':model.prediction_head.parameters(),'init_lr':args.lr, 'fix_lr': True}
            ] 

    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


    for ep in range(cfg.epoch):
        model.train()
        adjust_learning_rate(optimizer, args.lr, ep, args,args.lr_min)
        loss_avg = AverageMeter()
        loss_SSL = AverageMeter()
        loss_NEG = AverageMeter()
        for i, X in enumerate(trainloader):
            x, y = X
            x0,x1 = x[0],x[1]
            x0 = x0.cuda(non_blocking=True)
            x1 = x1.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            z0, p0 = model(x0)
            z1, p1 = model(x1)
         
            loss_ssl = 0.5 * (alignment_loss(p0,z1.detach(),y) + alignment_loss(p1,z0.detach(),y)) #
            loss_neg = uniform_loss(z0,z1,y,tmp=args.tmp) * args.lamb
            loss = loss_ssl + loss_neg 
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            loss_avg.update(loss.item(), x0.size(0))
            loss_SSL.update(loss_ssl.item(),x0.size(0))
            loss_NEG.update(loss_neg.item(),x0.size(0))

        tmm_lr= optimizer.param_groups[0]["lr"]
        print(f'Epoch {ep}: Loss {loss_avg.avg:.5f} POS {loss_SSL.avg:.5f}  NEG{loss_NEG.avg:.5f}, lr:{tmm_lr:.5f} ')
        logging.info(f'Epoch {ep}: Loss {loss_avg.avg:.5f} POS {loss_SSL.avg:.5f}  NEG{loss_NEG.avg:.5f}, lr:{tmm_lr:.5f}')

    acc = misc.net_pred_fast(model.backbone,args)    
    torch.save(model.backbone.state_dict(), f'{args.checkpoint_dir}/{cfg.encoder}_asycon.pth')
    logging.info('save model at ' + f'{args.checkpoint_dir}/{cfg.encoder}_asycon.pth')
    print('save model at ' + f'{args.checkpoint_dir}/{cfg.encoder}_asycon.pth')
    
def alignment_loss(p,z,y):
    y = y.view(-1,1)
    label_mask = torch.eq(y, y.T).float().cuda()
    p_norm = F.normalize(p,dim=-1)
    z_norm = F.normalize(z,dim=-1)
    cosine_similarity = torch.matmul(p_norm,z_norm.T)
    cosine_pos_mean = (cosine_similarity * label_mask).sum(1)/(label_mask.sum(1)+1e-6) #
    loss = -1 * cosine_pos_mean.mean()
    return loss

def uniform_loss(z0,z1,y,tmp=0.5):
    y = y.view(-1,1)
    label_mask = torch.eq(y, y.T).float()
    neg_num = len(y)- label_mask.sum(1)
    neg_mask = 1 - label_mask 
    z0_norm = F.normalize(z0,dim=-1)
    z1_norm = F.normalize(z1,dim=-1)
    neg = torch.matmul(z0_norm,z1_norm.T) * neg_mask
    loss = neg.div(tmp).exp().sum(1).div(neg_num).log().mean()
    return loss


if __name__ == '__main__':
    misc.print_args(cfg)
    main()
 