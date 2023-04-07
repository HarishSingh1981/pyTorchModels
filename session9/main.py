!git clone https://github.com/HarishSingh1981/pyTorchModels.git
%cd pyTorchModels/
!git pull origin
%cd ..
!pip install torchsummary
!pip install torch_lr_finder

'''Train CIFAR10 with PyTorch.'''
#!pip install grad-cam
import sys
sys.path.append('/content/pyTorchModels/')
sys.path.append('/content/pyTorchModels/models/session9')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import albumentations as Alb
import torchvision
import torchvision.transforms as transforms
import numpy as np
from utils import *
from testModel import *
from trainModel import *
from custom_VIT import *
#from utils import progress_bar
from albumentation_helper import *
import dataloader
from torch_lr_finder import LRFinder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 512

# Data
print('==> Preparing data..')

use_cuda = torch.cuda.is_available()

#kwargs_train = dict(num_workers= 4, pin_memory= True if use_cuda else False,shuffle=True,batch_size=batch_size)
#kwargs_test = dict(num_workers= 4, pin_memory= True if use_cuda else False,shuffle=False,batch_size=batch_size)

CustomTrainTransform = Alb.Compose([Alb.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                                    Alb.PadIfNeeded(min_height=36,min_width=36),
                                    Alb.RandomCrop(32,32,p=1),
                                    Alb.HorizontalFlip(p=1),
                                    Alb.CoarseDropout(max_holes=1,max_height=8,max_width=8,min_holes=1,min_height=8,min_width=8,fill_value=(0.5,0.5,0.5),mask_fill_value=None)])
 
transform = Alb_Transforms(CustomTrainTransform)
TestTransform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainloader = dataloader.getDataLoader(dataset=torchvision.datasets.CIFAR10, isTrain=True,transform=transform,batchSize=batch_size,isShuffle=True,workers=4,needDownload=True)
testloader = dataloader.getDataLoader(dataset=torchvision.datasets.CIFAR10, isTrain=False,transform=TestTransform,batchSize=batch_size,isShuffle=False,workers=4,needDownload=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Model
print('==> Building model..')
net = custom_VIT(0.025)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#summary(net, input_size=(3, 32, 32))

#define instance of LRFinder
criterion = nn.CrossEntropyLoss()
optimizer = optim.ADAM(net.parameters(), lr=0.001)

lr_finder = LRFinder(net,optimizer,criterion)

#Run range test
lr_finder.range_test(trainloader,start_lr=1e-03,end_lr=1,num_iter=100,step_mode="exp")
lr_finder.plot()
lr_finder.reset()

from torch.optim.lr_scheduler import OneCycleLR
'''
With above graph the fine tuned max lr is 0.0614 and as discussed earlier in class min LR 0.00614
that is what we used to calculate optimal LR
'''
ler_rate = 0.0614
#Let's start with loading the trained model and train again further
#model_parameters = torch.load('/content/pyTorchModels/custom_resnet.pth')
#load parameters to model
#net.load_state_dict(model_parameters)
criterion = nn.CrossEntropyLoss()
optimizer = optim.ADAM(net.parameters(), lr=0.001)

#Train again with one cycle policy now
epochs = 24
steps_per_epoch = len(trainloader) 
total_steps = epochs * len(trainloader)
#We need to achieve MAX LR in 5th epoch so calculate pct_start. Initial LR is max_lr/10
#lr has to go from max_lr/10 --> max_lr in 5 epochs that is 5*steps_per_epoch steps 
pct_start = (5*steps_per_epoch)/total_steps
print(f'pct_start --> {pct_start}')
scheduler = OneCycleLR(optimizer,max_lr=ler_rate,
                       steps_per_epoch=steps_per_epoch,epochs=epochs,
                       pct_start=pct_start,div_factor=10,final_div_factor=10,verbose=False)

#Train the model further
for epoch in range(start_epoch, start_epoch+24):
    train_model(dataloader=trainloader,network=net,lossfn=criterion,optimizer=optimizer,scheduler=scheduler)
    test_model(dataloader=testloader,network=net,lossfn=criterion)
