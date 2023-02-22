import albumentations as Alb
import numpy as np

class Alb_Transforms:
    def __init__(self, transforms: Alb.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        images = self.transforms(image=np.array(img))
        images = list(images.values())
        images = images[0]
        return images.permute(0,3,1,2)
