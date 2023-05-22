import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import random


class ContrastiveDataset(Dataset):

    def __init__(self, images_ds : Dataset, sketches_ds : Subset, transform : transforms = None):
        self.images_ds = images_ds
        self.sketches_ds = sketches_ds
        self.transform = transform
        self.sketches_dict = self.create_sketches_dict()


    def create_sketches_dict(self):
        class_indices = {}
        for idx, (_, label) in enumerate(self.sketches_ds):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices
        

    def __getitem__(self, index):
        image, image_class = self.images_ds[index]

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            idx = random.choice(self.sketches_dict[image_class])
        
        else:
            random_class = random.choice([idx for idx in list(self.sketches_dict.keys()) if idx != image_class])
            idx = random.choice(self.sketches_dict[random_class])
        
        if self.transform:
            image = self.transform(image)
        sketch, sketch_class = self.sketches_ds[idx]
        inputs = torch.stack((image, sketch))

        # Indicate whether a pair of samples (input1 and input2) are similar or dissimilar
        # 0: are the same class, 1: different class
        target = torch.tensor([(image_class != sketch_class)]).long()
        
        return inputs, target
    

    def __len__(self):
        return len(self.images_ds)


class AugmentedContrastiveDataset(ContrastiveDataset):
  def __init__(self, images_ds : Dataset, sketches_ds : Subset, transform : transforms = None):
    super().__init__(images_ds, sketches_ds, transform)

  def __getitem__(self, idx):
    input, target = super().__getitem__(idx)
    return torch.stack((self.transform(input[0]), self.transform(input[1]))), target


class TripletDataset(Dataset):

    def __init__(self, images_ds : Dataset, sketches_ds : Subset, transform : transforms = None):
        self.images_ds = images_ds
        self.sketches_ds = sketches_ds
        self.transform = transform
        self.sketches_dict = self.create_sketches_dict()
    

    def create_sketches_dict(self):
        class_indices = {}
        for idx, (_, label) in enumerate(self.sketches_ds):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices


    def __getitem__(self, index):
        image, image_class = self.images_ds[index]

        positive_idx = random.choice(self.sketches_dict[image_class])
        random_negative_class = random.choice([idx for idx in list(self.sketches_dict.keys()) if idx != image_class])
        negative_idx = random.choice(self.sketches_dict[random_negative_class])
          
        positive_sketch, _, = self.sketches_ds[positive_idx]
        negative_sketch, _ = self.sketches_ds[negative_idx]

        inputs = torch.stack((image, positive_sketch, negative_sketch))
        # Indicate whether a pair of samples (input1 and input2) are similar or dissimilar
        # 0: are the same class, 1: different class
        target = torch.tensor([0, 1])
        
        return inputs, target
    

    def __len__(self):
        return len(self.images_ds)
    

class AugmentedTripletDataset(TripletDataset):
  def __init__(self, images_ds : Dataset, sketches_ds : Subset, transform : transforms = None):
    super().__init__(images_ds, sketches_ds, transform)

  def __getitem__(self, idx):
    input, target = super().__getitem__(idx)
    return torch.stack((self.transform(input[0]), self.transform(input[1]), self.transform(input[2]))), target