import albumentations as Alb
import numpy as np
import torch

class Alb_Transforms:
    def __init__(self, transforms: Alb.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        images = self.transforms(image=np.array(img))
        images = list(images.values())
        images = images[0]
        images = images.transpose((2,0,1))
        #print(f'Shape of data from dataloader--> {images.shape}')
        return torch.from_numpy(images)
