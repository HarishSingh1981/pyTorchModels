import torch
import torchvision
import torchvision.transforms as transforms

use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

def getDataLoader(dataset,isTrain,transform,batchSize,isShuffle,workers,needDownload):
	kwargs = dict(num_workers= workers, pin_memory= True if use_cuda else False,shuffle=isShuffle,batch_size=batchSize)
	data_set = dataset(root='./data', train=isTrain,
                                        download=needDownload, transform=transform)
	dataloader = torch.utils.data.DataLoader(data_set, **kwargs)
	return dataloader