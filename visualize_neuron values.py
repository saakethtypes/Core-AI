import pathlib
import numpy as np
import torch 
import torch.nn.functional as F 
from torch.nn import Module, Linear
from torch.utils.tensorboard import SummaryWriter

class Network(Module):
    def __init__(self):
        super().__init__()

        self.fc_1 = Linear(15,20)
        self.fc_2 = Linear(20,30)
        self.fc_3 = Linear(30,2)
    
    def forward(self,x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = F.relu(x)
        return x 


if __name__ == '__main__':
    x = torch.rand(15)
    net = Network()

    #activation hook 
    writer = SummaryWriter()  
    def af_hook(inst,inp,out):
        writer.add_histogram(repr(inst),out)
    #registering hook per layer
    hook1 = net.fc_1.register_forward_hook(af_hook)
    net.fc_2.register_forward_hook(af_hook)
    net.fc_3.register_forward_hook(af_hook)
    
    #to remove a hook 
    hook1.remove()
    y = net(x)

#to visualize the histograms run - 
#tensorboard --logdir=runs 