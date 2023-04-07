# Custom resnet model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DepthwiseSeparable(nn.Module):
  def __init__(self, in_ch, out_ch, stride=1):
    super(DepthwiseSeparable, self).__init__()
    self.in_chan = in_ch
    self.out_chan = out_ch

    self.depthwise = nn.Sequential(
          nn.Conv2d(in_channels=self.in_chan, out_channels=self.in_chan, kernel_size=(3, 3), padding=1, stride=stride, groups=self.in_chan, bias=False),
          #pointwise
          nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(1,1)))

  def forward(self, x):
    x = self.depthwise(x)
    return x

class Ultimus(nn.Module):
  def __init__(self, in_ch, out_ch, stride=1):
    super(Ultimus, self).__init__()
    self.in_chan = in_ch
    self.out_chan = out_ch

    self.K = nn.Linear(in_features=48,out_features=8)
    self.Q = nn.Linear(in_features=48,out_features=8)
    self.V = nn.Linear(in_features=48,out_features=8)
    self.Z_Out = nn.Linear(in_features=8,out_features=48)
    
  def forward(self, x):
    X_K = self.K(x) #weighted Key
    X_Q = self.Q(x) #Weighted query
    X_V = self.V(x) #weighted value
    #print(f'Shape of X_K/X_Q/X_V --> {X_K.shape}/{X_Q.shape}/{X_V.shape}')
    AM = F.softmax(torch.matmul(X_K,torch.transpose(X_Q,0,1))/torch.sqrt(torch.tensor(X_Q.shape[1],dtype=torch.int32)))
    Z = torch.matmul(AM,X_V)
    return x + self.Z_Out(Z)

class custom_VIT(nn.Module):
    def __init__(self,drop):
        super(custom_VIT, self).__init__()
        '''
        j_out = j_in * stride
        nout = (n_in + 2*p-k)/s + 1
        rf_out = rf_in + (k-1)*j_in
        '''
        
        # Input: 32x32x3 | Output: 32x32x48 | RF: 7x7
        self.ultimus_blk = Ultimus(48,48)
        self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
            )
        # output = 1x1x48
        self.gap = nn.AvgPool2d(32)
        # Input: 32x32x64 | Output: 16x16x128 | RF: 3x3
        self.FC = nn.Linear(in_features=48,out_features=10)
    def forward(self, x):
      #maxpool must be used at least after 2 convolution and sud be as far as possible from last layer
        #x = x.to('cuda')
       # x = self.prepLayer(x)
        x = self.gap(self.prepLayer(x))
        x = x.view(-1,48)
        x = self.ultimus_blk(x)
        x = self.ultimus_blk(x)
        x = self.ultimus_blk(x)
        x = self.ultimus_blk(x)

        x = self.FC(x)    
        return F.log_softmax(x,dim=-1)
