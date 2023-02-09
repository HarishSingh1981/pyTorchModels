import albumentations as Alb
class Alb_Transforms:
    def __init__(self, transforms: Alb.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))
