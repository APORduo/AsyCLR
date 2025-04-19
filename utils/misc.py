import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import logging

from torch.distributions import Categorical
from torch.utils.data import DataLoader

from collections import defaultdict
from dataloader.data_utils import get_base_dataloader,get_new_dataloader
from dataloader import dataset



from sklearn.metrics import confusion_matrix,precision_score
from sklearn.metrics import pairwise
from sklearn.linear_model import SGDClassifier

from .metric import intra_dist,inter_cdist,inter_dist

dataset_proj={
    'cifar100':'cifar_cil',
    'mini_imagenet':'mini_cil',
    'cub200':'cub_cil',
}

def l2norm(x):
    return F.normalize(x,p=2,dim=1)
def top1_acc(logits, label):
    '''
    compute top1 accuarcy
    pred.shape  = (n_batch, n_class)
    label.shape = (n_batch,)
    '''
    if isinstance(logits, torch.Tensor):
        pred_y = logits.argmax(dim=1)
        acc = (pred_y == label).float().mean().item()
        return acc
    elif isinstance(logits, np.ndarray):
        pred_y = logits.argmax(axis=1)
        acc = (pred_y == label).mean()
        return acc

    return acc

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def cosine_similarity_tensor(x:torch.Tensor,y:torch.Tensor):
    x = F.normalize(x,p=2,dim=1)
    y = F.normalize(y,p=2,dim=1)
    return torch.mm(x,y.t())

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args,session)
    return trainset, trainloader, testloader


def get_logits( backbone, weight, scalar, x,return_feature = False):
        '''
        get the output logits based on the given feature extractor, classification weights and scalar factor
        '''
        f0 = backbone.forward(x)                      # (B, 3, H, W)
        f = F.normalize(f0, p=2, dim=1, eps=1e-12)                              # (B, C)
        weight = F.normalize(weight, p=2, dim=1, eps=1e-12)                     # (classes_seen, C)
        pred = scalar * torch.mm(f, weight.t())                                 # (B, classes_seen)
        
        if return_feature:
            return pred, f0
        return pred   
    

def get_cos_logits_byfeat(feats,weight,scalar) :
    logits = scalar * F.linear(l2norm(feats),l2norm(weight))
    return logits

def get_cos_logits( backbone, weight, scalar, x,return_feature = False):
        '''
        get the output logits based on the given feature extractor, classification weights and scalar factor
        '''
        f0 = backbone.forward(x)                      # (B, 3, H, W)
        f = F.normalize(f0, p=2, dim=1, eps=1e-12)                              # (B, C)
        weight = F.normalize(weight, p=2, dim=1, eps=1e-12)                     # (classes_seen, C)
        pred = scalar * torch.mm(f, weight.t())                                 # (B, classes_seen)
        
        if return_feature:
            return pred, f0
        return pred    
    

def get_all_feat_label(net,args):
    trainset_all = dataset.make(args.dataset)
    testset_all = dataset.make(args.dataset,train=False)

    trainlaoder_all = torch.utils.data.DataLoader(trainset_all,batch_size=128,shuffle=False,num_workers=8)
    testloader_all = torch.utils.data.DataLoader(testset_all,batch_size=128,shuffle=False,num_workers=8)

    trainset_all.transform = testloader_all.dataset.transform

    feat_train,label_train = extract_feature(net,trainlaoder_all)
    feat_test,label_test = extract_feature(net,testloader_all)
    return feat_train,label_train,feat_test,label_test

def get_binary_result(scores, y, session, n_base):
    if isinstance(scores,np.ndarray):
        scores = torch.from_numpy(scores)
        y = torch.from_numpy(y)
        
    mask_base  = (y <  n_base)
    mask_novel = (y >= n_base)
    pred_b = scores[mask_base]              # (N_b, n_way_all)
    pred_n = scores[mask_novel]             # (N_n, n_way_all)
    # joint accuracy
    _, pred_a = torch.max(scores, dim=1)    # (N, )
    pred_a = (pred_a < n_base)
    acc_joint = (pred_a == mask_base).float().mean().item()
    # base class samples            
    _, pred_b = torch.max(pred_b, dim=1)    # (N_b, ) base samples
    pred_b2b = (pred_b <  n_base).float()
    pred_b2n = (pred_b >= n_base).float()
    acc_b2b, N_b2b, N_b2n = pred_b2b.mean().item(), pred_b2b.sum().item(), pred_b2n.sum().item()
    # novel class samples
    _, pred_n = torch.max(pred_n, dim=1)    # (N_n, ) novel samples
    pred_n2n = (pred_n >= n_base).float()
    pred_n2b = (pred_n <  n_base).float()
    acc_n2n, N_n2n, N_n2b = pred_n2n.mean().item(), pred_n2n.sum().item(), pred_n2b.sum().item()
    str_out = '[Binary CLS Results: %.2f%%] Session %d acc:b2b=%.2f%% num: b2b=%d b2n=%d; acc: n2n=%.2f%%, num: n2n=%d, n2b=%d'%(100*acc_joint, session, 
                            100*acc_b2b, N_b2b, N_b2n, 100*acc_n2n, N_n2n, N_n2b)
    return str_out

def get_detailed_result(scores, y, session,args ,str_out=''):
        assert session > 0
        base_way = args.base_class
        c2a_list = []
        c2c_list = []
        for t in range(session + 1):
            if t == 0:
                c_start, c_end = 0, base_way - 1
            else:
                c_start = base_way + (t - 1) * args.n_way
                c_end   = base_way + t * args.n_way - 1
            idx_tmp  = (y >= c_start) & (y <= c_end)      
            y_all    = y[idx_tmp]                       # (n_test, )
            y_tmp    = y_all - c_start                  # (n_test, )
            pred_all = scores[idx_tmp]                  # (n_test, n_way_all)
            
            pred_tmp = pred_all[:, c_start: c_end+1]    # (n_test, n_way)
            c2a = top1_acc(pred_all, y_all)
      
            c2c = top1_acc(pred_tmp, y_tmp)
            c2a_list.append(c2a)
            c2c_list.append(c2c)
            str_out += ' |%.2f%%, %.2f%%|'%(100*c2c, 100*c2a)
        
        all_acc = top1_acc(scores, y)
        str_out = f'{all_acc:.2%}'+str_out
        return str_out, c2a_list, c2c_list
def get_novel_result(scores, y,args):
    base_class = args.base_class
    novel_index = (y >= base_class)
    whole_acc = top1_acc(scores[novel_index], y[novel_index])
    oracle_acc = top1_acc(scores[novel_index,base_class:], y[novel_index]-base_class)
    
    str_novel = 'whole acc: %.2f%%, oracle acc: %.2f%%'%(100*whole_acc, 100*oracle_acc)
    return str_novel
def get_class_mean(backbone, trainloader, train_set=None,norm_first = True):
    '''
    get the ordered class prototypes from trainloader based on the given feature extractor
    '''
    if train_set is None:
        train_set = trainloader.dataset
    
    if isinstance(backbone,nn.Module):
        backbone.eval()
    data = []
    label = []
    class_list = np.unique(train_set.targets).tolist()
    class_list.sort()
    with torch.no_grad():
        for X in trainloader:
            x = X[0]
            y = X[1]
            if isinstance(x,list):
                x = torch.cat(x,dim=0)
                y = torch.cat(y,dim=0)
            data_tmp, label_tmp = x.cuda(),y.cuda()
            data_tmp = backbone(data_tmp)       
            data.append(data_tmp)
            label.append(label_tmp)
        data = torch.cat(data, dim=0)
        label = torch.cat(label, dim=0)
        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]                    
            if norm_first:
                embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)#norm_first
            proto = embedding.mean(0)                   
            new_fc.append(proto)
        new_fc = torch.stack(new_fc, dim=0)
    return new_fc

def get_novel_class_mean(backbone, trainloader, train_set,norm_first = True):

    backbone.eval()
    data = []
    label = []
    class_list = np.unique(train_set.targets).tolist()
    class_list.sort()
    with torch.no_grad():
        for x,y in trainloader:
            if isinstance(x,list):
                x = torch.cat(x,dim=0)
                y = torch.cat(y,dim=0)
            data_tmp, label_tmp = x.cuda(),y.cuda()
            data_tmp = backbone.part_forward(data_tmp)
                   
            data.append(data_tmp)
            label.append(label_tmp)
        data = torch.cat(data, dim=0)
        label = torch.cat(label, dim=0)
        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            z = data[data_index] 
            z_avg = z.mean(0,keepdim=True)
            embedding = backbone.last_forward(z_avg)                   
            if norm_first:
                embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)#norm_first
            proto = embedding.squeeze(0)                  
            new_fc.append(proto)
        new_fc = torch.stack(new_fc, dim=0)
    return new_fc
    
def get_class_mean_byfeat(feats,label):
    if isinstance(label,torch.Tensor):
        label = label.cpu().data.numpy()
    class_list = np.unique(label).tolist()
    label = np.array(label)
    new_fc = []
    for class_index in class_list:
        data_index = (label == class_index)
        embedding = feats[data_index]                    
        proto = embedding.mean(0)                   
        new_fc.append(proto)
 
    # for loop ending
    if isinstance(feats,torch.Tensor):
        new_fc = torch.stack(new_fc, dim=0)
    else:
        new_fc = np.stack(new_fc, axis=0)
    return new_fc
        
def get_acc_by_feat_feat(train_feat, train_label,test_feat,test_label):
    if isinstance(train_feat,torch.Tensor):
        train_feat = train_feat.cpu().data.numpy()
    
    if isinstance(test_feat,torch.Tensor):
        test_feat = test_feat.cpu().data.numpy()
        
    if isinstance(train_label,torch.Tensor):
        train_label = train_label.cpu().data.numpy()
    
    if isinstance(test_label,torch.Tensor):
        test_label = test_label.cpu().data.numpy()
    
    tmp_train_label = train_label - min(train_label)
    tmp_test_label =   test_label - min(test_label)
    class_mean = get_class_mean_byfeat(train_feat,tmp_train_label)
    logits = pairwise.cosine_similarity(test_feat,class_mean)
    
    acc = (logits.argmax(1) == tmp_test_label).mean()

    return acc
    
    

def test_one_session(testloader, backbone, weight,scale_cls, args, session, prefix=None, report_acc=True, report_binary=False):
    '''
    evalute on the current session t based on the given feature extractor and classification weights
    '''
    tqdm_gen = tqdm.tqdm(testloader)
    if isinstance(backbone,nn.Module):
        backbone.eval()
    y = []
    feats = []
    scores = []

    assert (args.base_class + session * args.n_way) == weight.shape[0]
    with torch.no_grad():
        for i, batch in enumerate(tqdm_gen):
            data, test_label = [_.cuda() for _ in batch]
            feat = backbone(data)
            #feat_norm = F.normalize(feat,dim=1)
            if isinstance(weight,nn.Module):
                _score = weight(feat)
            else:
                weight_norm = F.normalize(weight,dim=1)
                _score = F.linear(feat,weight_norm)
            scores.append(_score)  
            y.append(test_label)  
            feats.append(feat)       
    y = torch.cat(y, dim=0)
    scores = torch.cat(scores, dim=0)
    feats = torch.cat(feats,dim=0)
    scores = F.softmax(scale_cls*scores,-1)
    acc = top1_acc(scores, y)
    
    base_way = args.base_class
    if session > 0 and report_binary == True:
        str_out = get_binary_result(scores, y, session, base_way)
        print(str_out)
        logging.info(str_out)

    if report_acc == True:
        str_out = '' if prefix is None else prefix
        str_out += f'Session {session} testing accuracy = {acc:.2%}'
        if session > 0:
            str_out, c2a_list, c2c_list = get_detailed_result(scores, y, session, args,str_out)
        print(str_out)
        logging.info(str_out) 

    return acc, scores, feats, y


    
    

def test_one_session_simple(testloader, backbone, weight,return_score=False):
    tqdm_gen = tqdm.tqdm(testloader)
    if isinstance(backbone,nn.Module):
        backbone.eval()
    y = []
    scores =[]
    with torch.no_grad():
        for i, batch in enumerate(tqdm_gen):
            data, test_label = [_.cuda() for _ in batch]
            feat = backbone(data)
            feat_norm = F.normalize(feat,dim=-1)
            weight_norm = F.normalize(weight,dim=-1)
            _score = F.linear(feat_norm ,weight_norm)
            scores.append(_score)  
            y.append(test_label)  
            #feats.append(feat)       
    y = torch.cat(y, dim=0)
    scores = torch.cat(scores, dim=0)

    acc = top1_acc(scores, y)
    if return_score:
        return acc, scores
    return acc
    
def compute_accuracy(model,loader):
    model.eval()
    correct,total = 0
    for i,(x,y) in enumerate(loader):
        x,y = x.cuda(),y.cuda()
        with torch.no_grad():
            pred = model(x)
        correct += (pred.argmax(dim=-1) == y).float().sum().item()
        total += x.shape[0]

def test_supervised(backbone,head,loader,return_score = False):
    backbone.eval()
    head.eval()
    labels = []
    #feats = []
    scores = []
    with torch.no_grad():
        total = 0
        correct = 0
        t = tqdm.tqdm(loader, ncols=80)
        for (x, y) in t:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            _feat = backbone(x)
            _score = head(_feat)
            #pdb.set_trace()
            pred = _score.argmax(dim=-1)
            correct += (pred == y).float().sum().item()
            total += x.shape[0]

            acc = correct / total
            #t.set_description(f'acc = {acc:4.2%}')
            t.set_postfix(acc=f'{acc:4.2%}')
            labels.append(y)
            #feats.append(_feat)
            scores.append(_score)
    
    #print(f'Test supervised accuracy: {acc:4.2%}')
    acc = correct / total
    if return_score:
        return acc,scores,labels

    return acc


    
class Loss(object):
    def __init__(self) -> None:
        pass
        
    
    def sum(self):
        loss = 0
        #pdb.set_trace()
        for k,v in self.__dict__.items():
            loss += v
        
        return loss
    
    def __repr__(self) -> str:
        info = "Loss--> "
        for k,v in self.__dict__.items():
            if v>0:
                info += f" {k}:{float(v):.4f}"
        return info
    

def extract_feature(backbone, loader,return_dict = False,return_tensor = False):
    backbone.cuda()
    backbone.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for (x, y) in tqdm.tqdm(loader, desc='extracting feature', ncols=80):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            feats = backbone(x)
            if feats.dim() > 2:
                feats = feats.squeeze()
            elif feats.dim() == 1:
                feats = feats.unsqueeze(0)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if return_tensor:
        all_feats  = torch.from_numpy(all_feats).cuda()
        all_labels = torch.from_numpy(all_labels).cuda()
    if return_dict:
        cl_data_file = defaultdict(list)
        for feat, label in zip(all_feats, all_labels):
            cl_data_file[label].append(feat)
        return cl_data_file
    else:
        return all_feats,all_labels
    
def get_lcs_acc(feat_train,label_train,feat_test,label_test):
    linear_clf = SGDClassifier(
            fit_intercept=False,
            ).fit(feat_train, label_train)
    acc = linear_clf.score(feat_test, label_test)
    return acc

def topk(mat,k):
    if isinstance(mat,np.ndarray):
        mat = torch.from_numpy(mat)
    
    return torch.topk(mat,k=k,dim=-1)


def energy(logit,tmp=1):
    if isinstance(logit,np.ndarray):
        logit = torch.from_numpy(logit)
        
    return  -1 *  (tmp * torch.logsumexp(logit / tmp, dim=1))

def net_pred(net,args,return_class_acc = False,norm_first = True):
    net.eval()
    opt = args
    acc_list = []
    n_session = opt.n_sessions
    train_set ,train_loader,test_loader0 = get_dataloader(opt,session=0)
    train_set.transform = test_loader0.dataset.transform
    weight_simple = get_class_mean(net,train_loader,train_set ,norm_first=norm_first)
    acc1, scores1, feats1, y1 = test_one_session(test_loader0,net,weight_simple,16,args=opt,session=0,report_acc=True,report_binary=True)
    
    

    acc_list.append(acc1)
    for session in range(1,n_session):
        train_set ,train_loader,test_loader = get_dataloader(opt,session=session)
        train_set.transform = test_loader.dataset.transform
        weight_simple_tmp = get_class_mean(net,train_loader,train_set ,norm_first=norm_first)
        weight_simple = torch.cat([weight_simple,weight_simple_tmp],dim=0)
        acc1, scores1, feats1, y1 = test_one_session(test_loader,net,weight_simple,16,args=opt,session=session,report_acc=True,report_binary=True)
   
        acc_list.append(acc1)
    
    # last session confusion matrix
    y_pred = scores1.argmax(dim=-1)
    con_mat = confusion_matrix(y1.cpu().numpy(), y_pred.cpu().numpy(),normalize='true')
    class_acc = np.diag(con_mat)

    print('NCM:',accstr(acc_list))
    if return_class_acc:
        return class_acc
    return accstr(acc_list)

def net_pred_fast(net,args,transform = None,return_logits = False):
    dataset_cil = dataset_proj[args.dataset]
    trainset = dataset.make(dataset_cil,train=True)
    testset = dataset.make(dataset_cil,train=False)
    if transform is not None:
        trainset.transform = transform
        testset.transform = transform
    else:
        trainset.transform = testset.transform
    trainloader =  DataLoader(trainset,batch_size=128,shuffle=True,num_workers=4)
    testloader = DataLoader(testset,batch_size=128,shuffle=False,num_workers=4)
    
    feat,label = extract_feature(net,trainloader)
    base_idndex = (label < args.base_class)
    feat_norm = feat / np.linalg.norm(feat,axis=1,keepdims=True)
    class_mean_norm = get_class_mean_byfeat(feat_norm,label)
    feat_test,label_test = extract_feature(net,testloader)

    logits_norm = pairwise.cosine_similarity(feat_test,class_mean_norm)

    acc_norm = top1_acc(logits_norm, label_test)

    
    str_binay_norm = get_binary_result(logits_norm,label_test,args.n_sessions-1,args.base_class)
    str_out_norm = get_detailed_result(logits_norm,label_test,session=args.n_sessions-1,args=args)[0]
    str_novel_norm = get_novel_result(logits_norm,label_test,args)

    
    logging.info('Normed Feature ' +str_out_norm)
    logging.info(str_binay_norm)
    logging.info('Normed' +str_novel_norm)
    print('Normed Feature ' + str_out_norm)
    print(str_binay_norm)
    print(str_novel_norm)
    if return_logits:
        return logits_norm,label_test
    return acc_norm
def accstr(acc_list:list):
    sl = []
    for n in acc_list:
        n = float(n)
        sl.append (f'{n:.3%}')
    
    sout = str(sl).replace("'","")
    return sout
        
def test_NCM_session(net,train_loader,test_loader,session,args,weight_prev= None):
    net.eval()
    opt = args
    train_set = train_loader.dataset
    train_set.transform = test_loader.dataset.transform
    weight = get_class_mean(net,train_loader,train_set ,norm_first=args.norm_first)
    if weight_prev is not None:
        weight = torch.cat([weight_prev,weight],dim=0)
    acc1, scores1, feats1, y1 = test_one_session(test_loader,net,weight,16,args=opt,session=session,report_acc=True,report_binary=True,prefix='NCM')

    return acc1,weight

def print_args(args):
    try:
        #pdb.set_trace()
        for key, value in sorted(args.items()):
            logging.info("{}: {}".format(key, value))
    except:
        for key, value in sorted(vars(args).items()):
            logging.info("{}: {}".format(key, value))
        
    