import albumentations as Alb
import numpy as np

class Alb_Transforms:
    def __init__(self, transforms: Alb.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        inputs = self.transforms(image=np.array(img))
        inputs = list(inputs.values())
	inputs = inputs[0]
        return inputs
