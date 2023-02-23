!git clone https://github.com/HarishSingh1981/pyTorchModels.git
%cd pyTorchModels/
!git pull origin
%cd ..
!pip install torchsummary
'''Train CIFAR10 with PyTorch.'''
!pip install grad-cam
import sys
sys.path.append('/content/pyTorchModels/')
sys.path.append('/content/pyTorchModels/models/session8')
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
from custom_resnet import *
#from utils import progress_bar
from albumentation_helper import *
import dataloader
#working on GradCAM 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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
visualTransform = transforms.Compose([transforms.ToTensor()])
#why we need normalized test data
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
trainloader = dataloader.getDataLoader(dataset=torchvision.datasets.CIFAR10, isTrain=True,transform=transform,batchSize=batch_size,isShuffle=True,workers=4,needDownload=True)
testloader = dataloader.getDataLoader(dataset=torchvision.datasets.CIFAR10, isTrain=False,transform=TestTransform,batchSize=batch_size,isShuffle=False,workers=4,needDownload=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#visualize the 20 images from train set
vis_dataloader = dataloader.getDataLoader(dataset=torchvision.datasets.CIFAR10, isTrain=True,transform=visualTransform,batchSize=batch_size,isShuffle=True,workers=4,needDownload=True)

'''
def viz_data(customDataLoader,cols=8, rows=5):
  figure = plt.figure(figsize=(10, 10),dpi=64)
  dataiter = iter(customDataLoader)
  image,label = next(dataiter)
  #image = list(image.values())
  #images = image[0]

  #print(f'shape of image {image.shape}')
  #print(f'shape of label {label.shape}')

  for i in range(1, (cols * rows) + 1):
    figure.add_subplot(rows, cols, i)
    #print(f'label value {label[i-1].abs()}')
    plt.title(classes[label[i-1]])
    plt.axis("off")
    plt.imshow(image[i-1].permute(1,2,0))

  plt.tight_layout()
  plt.show()

viz_data(vis_dataloader,4,5)
'''
# Model
print('==> Building model..')
net = custom_resnet(0.025)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#summary(net, input_size=(3, 32, 32))

'''
# Print the names of the layers in the network
for name, layer in net.named_modules():
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print(f'Name  --> {name}')
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


print(net)

#Visualization of data
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
#print(f'images data type {type(images)} and type of label {type(labels)}')
#images = list(images.values())
#print(f'images data type {type(images)} and type of label {type(labels)} and size of image {len(images)} and shape of image {images[0].shape}')
#images = images[0]
#print(f'images data type {type(images)} and type of label {type(labels)}')

# show images
imshow_cifar(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(start_epoch, start_epoch+20):
    train_model(dataloader=trainloader,network=net,lossfn=criterion,optimizer=optimizer)
    test_model(dataloader=testloader,network=net,lossfn=criterion)
#    scheduler.step()

class_correct = list(0. for i in range(10))
wrong_image_list = []
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
      images, labels = data
      #print(f"Shape of labels --> {labels.shape} and type {type(labels)}")
      if torch.cuda.is_available():
      #     images = list(images.values())
      #     images = images[0]
        images = images.to('cuda')
        labels = labels.to('cuda')
      outputs = net(images.permute(0,1,2,3))
      #collecting indices where max happened in predicted
      _, predicted = torch.max(outputs, 1)
      #print(f'predicted shape {predicted}')
      c = (predicted == labels).squeeze()
      for i in range(int(labels.size()[0])):
          label = labels[i]
          class_correct[label] += c[i].item()
          class_total[label] += 1
          if c[i].item() == 0:
            wrong_image_list.append((images[i],predicted[i]))


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

#save the model in a file
torch.save(net.state_dict(),'/content/pyTorchModels/custom_resnet.pth')
#find out the 10 failed images

figure = plt.figure(figsize=(10, 10),dpi=64)
for i in range(1, (5 * 2) + 1):
  figure.add_subplot(2, 5, i)
  image,label = wrong_image_list[i]
  image = image.to('cpu')
  plt.title(classes[label])
  plt.axis("off")
  plt.imshow(image.permute(1,2,0))

plt.tight_layout()
plt.show()

#Apply grad cam
net = nn.DataParallel(net)
gdcam_dataloader = dataloader.getDataLoader(dataset=torchvision.datasets.CIFAR10, isTrain=False,transform=visualTransform,batchSize=10,isShuffle=True,workers=4,needDownload=True)
iterator = iter(gdcam_dataloader)
#get next 10 images
images , labels = next(iterator)
target_layer = [net.module.convblock4[2]]
#input_tensor = # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=net, target_layers=target_layer, use_cuda=use_cuda)
output = net(images)
predictions = torch.argmax(output)

#compute the GradCam for the predicted class labels
print(f'Shape of images --> {images.shape}')
grayscale_cam = cam(images,target_layer)
#heatmap = cam.generate(target_class=predictions)
visualization = show_cam_on_image(images,grayscale_cam, use_rgb=True)

#find out the 10 failed images

figure = plt.figure(figsize=(10, 10),dpi=64)
for i in range(1, (5 * 2) + 1):
  figure.add_subplot(2, 5, i)
  image,label = wrong_image_list[i]
  image = image.to('cpu')
  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
  grayscale_cam = cam(input_tensor=image.unsqueeze(dim=0), targets=targets)

  # In this example grayscale_cam has only one image in the batch:
  grayscale_cam = grayscale_cam[0, :]
  

  plt.title(classes[label])
  plt.axis("off")
  plt.imshow(visualization.permute(1,2,0))

plt.tight_layout()
plt.show()

my_images = []
for i in range(10):
    image = images[i]
    my_images.append(image)
my_images = torch.stack(my_images)

# Create an instance of the GradCAM class
cam = GradCAM(model=net, target_layers=target_layer, use_cuda=use_cuda)
# Generate the Grad-CAM heatmaps
target_classes = [5] * 10
heatmaps = []
for i in range(batch_size):
    mask = cam(images[i:i+1], targets=None)
    heatmaps.append(mask)
heatmaps = torch.cat(heatmaps, dim=0)

# Display the heatmaps in a matplotlib grid
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].permute(1, 2, 0))
    ax.imshow(heatmaps[i], alpha=0.5, cmap='jet')
    ax.axis('off')
plt.show()
'''
!pip install torch_lr_finder

from torch_lr_finder import LRFinder

#define instance of LRFinder
criterion_new = nn.CrossEntropyLoss()
optimizer_new = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

lr_finder = LRFinder(net,optimizer_new,criterion_new)

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
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
