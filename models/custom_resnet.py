# Custom resnet model
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

class custom_resnet(nn.Module):
    def __init__(self,drop):
        super(custom_resnet, self).__init__()
        '''
        j_out = j_in * stride
        nout = (n_in + 2*p-k)/s + 1
        rf_out = rf_in + (k-1)*j_in
        '''
        
        # Input: 32x32x3 | Output: 32x32x64 | RF: 3x3
        self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())
        # Input: 32x32x64 | Output: 16x16x128 | RF: 3x3
        self.layer1_X = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1,stride=1,bias=False), # Input: 32x32x64 | Output: 32x32x128 | RF: 5x5 
            nn.MaxPool2d(kernel_size=(2,2),stride=2), # Input: 32x32x128 | Output: 16x16x128 | RF: 6x6 | jin = 2
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer1_R1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1,bias=False), # Input: 16x16x128 | Output: 16x16x128 | RF: 10x10 | jin = 2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1,bias=False),# Input: 14x14x128 | Output: 16x16x128 | RF: 14x14 | jin = 2
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1,bias=False),  # Input: 16x16x128 | Output: 16x16x256 | RF: 18x18 | jin = 2
            nn.MaxPool2d(kernel_size=(2,2),stride=2), # Input: 16x16x256 | Output: 8x8x256 | RF: 22x22 | jin = 4
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer3_X = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1,stride=1,bias=False),  # Input: 8x8x256 | Output: 8x8x512 | RF: 30x30 | jin = 4
            nn.MaxPool2d(kernel_size=(2,2),stride=2),# Input: 8x8x256 | Output: 4x4x256 | RF: 22x22 | jin = 4
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer3_R1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,bias=False), # Input: 4x4x256 | Output: 4x4x512 | RF: 30x30 | jin = 4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),padding=1,bias=False), # Input: 4x4x512 | Output: 4x4x512 | RF: 38x38 | jin = 4
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.MP4 = nn.MaxPool2d(kernel_size=(4,4),stride=4) # Input: 4x4x512 | Output: 1x1x512 | RF: 42x42 | jin = 8
        self.fc = nn.Linear(in_features=512,out_feature=10)
    def forward(self, x):
      #maxpool must be used at least after 2 convolution and sud be as far as possible from last layer
        #x = x.to('cuda')
        x = self.prepLayer(x)
        x = self.layer1_X(x)
        res = self.layer1_R1(x)
        x = x+res
        x = self.layer2(x)
        x = self.layer3_X(x)
        res = self.layer3_R1(x)
        x = x+res
        x =  self.MP4(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=-1)
