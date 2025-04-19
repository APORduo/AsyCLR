import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))

        if sigma:
            self.sigma = 16.0
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
       

    def forward(self, input,*args):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out

        return out