


import torch
import torch.nn.functional as F
import logging
import copy 
from tqdm import tqdm


from utils.linears import CosineLinear
from utils.config import cfg,unknow
from utils import misc,util


from torchvision.models import resnet18
args = cfg
args.trainer = "FT"
args.epoch = 20 
args.model_path = ''
args.w_dis = 2.0
args.loss_type = 'cos'
args.lr = 1e-3
args.prefix = ''
args.trans = 'train'
args.imp = False
args.lr_scale = 2.0

args.merge_from_list(unknow) 
util.set_logger(args,prefix=args.loss_type)

model =resnet18().cuda()
model.feat_dim = 512
del model.fc 
model.load_state_dict(torch.load(args.model_path))
old_model = copy.deepcopy(model)
old_model.requires_grad_(False)


clf = CosineLinear(model.feat_dim,args.base_class).cuda()

params = [
    {'params':clf.parameters(),'lr':args.lr*args.lr_scale},
   {'params':model.parameters(),'lr':args.lr},

]

optim = torch.optim.SGD(params,lr = args.lr,momentum=0.9,weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=args.epoch,eta_min=5e-6)
trainset,trainloader,testloader = misc.get_base_dataloader(args)
trainset.transform = trainset.fine_transform

def main():
    for ep in range(args.epoch):
        tqdm_gen = tqdm(trainloader)
        old_model.eval()
        model.train()
        model.cuda()
        for i,(x,y) in enumerate(tqdm_gen):
            #x = torch.cat(x,dim=0) #two crop
            
            x = x.cuda()
            y = y.cuda()
            with torch.no_grad():
                old_feat = old_model(x)
            
            feat = model(x)
            logits = clf(feat)
            loss_ce =  F.cross_entropy(logits,y)
            loss_distill = match_loss(feat,old_feat,args.loss_type)
            loss = loss_ce + loss_distill * args.w_dis
            optim.zero_grad()
            loss.backward()
            optim.step()
            tmp_lr = optim.param_groups[0]["lr"]
            tqdm_gen.set_description(f'Epoch: {ep} | Iter: {i} | Loss: {loss.item():.4f},lr:{tmp_lr:.5f} ')

        lr_scheduler.step()
      
        misc.test_supervised(model,clf,testloader)

    misc.net_pred(model,args)
    torch.save(model.state_dict(), f'{args.checkpoint_dir}/{cfg.encoder}_ft.pth')
    logging.info('save model at ' + f'{args.checkpoint_dir}/{cfg.encoder}_ft.pth')
    
def match_loss(f1, f2,loss_type='cos'):
    #f1
    #f2 form old model
    # Compute the loss according to the loss type
    if loss_type == 'mse':
        loss = F.mse_loss(f1, f2)
    elif loss_type == 'rmse':
        loss = F.mse_loss(f1, f2) ** 0.5
    elif loss_type == 'mfro':
        # Mean of Frobenius norm, normalized by the number of elements
        loss = torch.mean(torch.frobenius_norm(f1 - f2, dim=-1)) / (float(f1.shape[-1]) ** 0.5)
    elif loss_type == "cos":
        loss = 1 - F.cosine_similarity(f1, f2, dim=1).mean()
    elif loss_type =='turkey':
        
        f2_turkey = torch.sign(f2) * (torch.abs(f2) ** 0.5)
        loss = 1 - F.cosine_similarity(f1, f2_turkey, dim=1).mean()
    elif loss_type=='fkd':
        teach_out = F.softmax(f2 /0.5, dim=-1)
        student_out = f1
        loss = torch.sum(-teach_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()

    elif loss_type == "zero":
        loss = 0
    else:
        raise ValueError("Unknown loss type: {}".format(loss_type))
    return loss

if __name__ == '__main__':
    main()