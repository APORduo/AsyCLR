import logging 
import os
def set_logger(args,log=True,prefix = '',trainer= None,logname = None):
    args.checkpoint_dir = "{}/{}/{}".format(
                            args.exp_dir,
                            args.dataset,
                            trainer or args.trainer,
                              )
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if log:
        if logname is None:
            logname =  f"{args.trainer.replace('/','')}_{args.schedule}_{args.encoder}_{args.epoch}_Head{args.head_type}_Loss{args.loss_type}.log"    

        logname = prefix + logname
        
        logging.basicConfig(filename=os.path.join(args.checkpoint_dir, logname), level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%d %b %Y %H:%M:%S')
        logging.info(str(os.sys.argv).replace("'","").replace(",",""))
        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
    

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def get_info(self):
        entries = [self.prefix] +  [str(meter) for meter in self.meters]
        return (' '.join(entries))
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]