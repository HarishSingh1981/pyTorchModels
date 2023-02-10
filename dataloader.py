import torch
import torchvision
import torchvision.transforms as transforms

use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

kwargs_train = dict(num_workers= 4, pin_memory= True if use_cuda else False,shuffle=True,batch_size=batch_size)


def getDataLoader(dataset,isTrain=True,transform,batchSize,isShuffle=True,workers=4,needDownload=True):
	kwargs = dict(num_workers= workers, pin_memory= True if use_cuda else False,shuffle=isShuffle,batch_size=batchSize)
	data_set = dataset(root='./data', train=isTrain,
                                        download=needDownload, transform=transform)
	dataloader = torch.utils.data.DataLoader(data_set, **kwargs_train)
	return dataloader