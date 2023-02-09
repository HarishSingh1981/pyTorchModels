!pip install torchsummary
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Net_Cifar10(nn.Module):
    def __init__(self):
        super(Net_Cifar10, self).__init__()
        '''
        j_out = j_in * stride
        nout = (n_in + 2*p-k)/s + 1
        rf_out = rf_in + (k-1)*j_in
        '''
        #input 3x32x32 -? OUtput 6x28x28? RF 5
        self.c1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5),
                                      nn.BatchNorm2d(6),
                                      nn.ReLU())
                                      #nn.Dropout2d(0.5))
        #input 6x28x28 -? OUtput 6x20x20? RF 13
        self.transitionBlock1 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5,dilation=2),
                                      nn.BatchNorm2d(6),
                                      nn.ReLU())
        #input 6x20x20 -? OUtput 16x20x20? RF 17
        self.c2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5,padding=2),
                                      nn.BatchNorm2d(12),
                                      nn.ReLU())
                                      #nn.Dropout2d(0.5))
        #input 12x20x20 -? OUtput 12x12x12? RF 33
        #self.transitionBlock1 = nn.MaxPool2d(2,2)
        self.transitionBlock2 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5,dilation=4,padding=4),
                                      nn.BatchNorm2d(12),
                                      nn.ReLU())

        #input 12x12x12 -? OUtput 16x8x8? RF 37
        self.c3 = nn.Sequential(nn.Conv2d(12,16,5),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU())
                                      #nn.Dropout2d(0.5))

        #input 16x8x8 -? OUtput 10x4x4? RF 45
        self.c4 = nn.Sequential(nn.Conv2d(16,16,5,groups=16),
                                nn.Conv2d(16,10,1),
                                nn.BatchNorm2d(10),
                                nn.ReLU())
                                #nn.Dropout2d(0.5))               
        self.avgPoolblk = nn.AvgPool2d(4,4)
        self.fc = nn.Linear(12,10)
    def forward(self, x):
      #maxpool must be used at least after 2 convolution and sud be as far as possible from last layer
        #x = x.to('cuda')
        x = self.c1(x)
        x = self.transitionBlock1(x)
        x = self.c2(x)
        x = self.transitionBlock2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.avgPoolblk(x)
    #    print(f'shape of x after GAP is {x.shape}')
        x = x.view(-1, 10)
       # x = self.fc(x)
        return x

if torch.cuda.is_available():
  my_model = My_Net().to('cuda')
else :
  my_model = My_Net()
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
#model = Net().to(device)
summary(my_model, input_size=(3, 32, 32))