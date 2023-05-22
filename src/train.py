import torch
from torch.optim import Adam
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split

from Utils import *
from Datasets import *
from Networks import *
from Losses import *
from EarlyStopper import *
from TrainingFunctions import training_loop

'''
SCRIPT DESCRIPTION:

This script is responsible for training a neural network that performs base information retrieval using sketches. 
Afterwards, you can modify certain configurations.
'''

dataset_paths = {'mini' : ["../Mini Dataset/photo", "../Mini Dataset/sketch"], 
                 'full' : ["../Full Dataset/256x256/photo", "../Full Dataset/256x256/sketch"]}





# ====================================
#               CONFIG
# ====================================

#Pick a Dataset (you can use the dictionary up here as reference)
#   Note that the dataset should be partitioned in folder readable by the ImageFolder class
#   You can download the dataset here: https://sketchy.eye.gatech.edu/
DATASET_NAME = 'mini'
PHOTO_DATASET_PATH, SKETCHES_DATASET_PATH = dataset_paths[DATASET_NAME]

#Pick a Dataset Type 
#   For training: ContrastiveDataset, TripletDataset, AugmentedContrastiveDataset, AugmentedTripletDataset
#   For validation: ContrastiveDataset, TripletDataset
#   IMPORTANT: If you are using an Augmented dataset remember to assing Composed Trasformations to TRANSFORMATION
TRAIN_DATASET_TYPE = ContrastiveDataset
TRANSFORMATION = None
VAL_DATASET_TYPE = ContrastiveDataset

#Pick a Criterion
#   This criterion MUST coincide with the previous dataset type
#   You can chose between: ContrastiveLoss, TripletLoss
CRITERION = ContrastiveLoss

#Pick an embedding size
#   Within the embedding space, we have numerous vectors, each with this dimension
#   If a higher level of detail is required, increase this value
#   In case of memory constraints, decrease it
OUTPUT_EMBEDDING = 16

#Choose a Weight Path
#   After the training your weight are going to be saved here
WEIGHT_PATH = f"../weights/{DATASET_NAME}-{OUTPUT_EMBEDDING}-contrastive-resnet50.pth"

#Pick a Margin
#   This is the input value provided to the contrastive loss and triplet loss
#   represents the proximity required between values to be associated with the same class
#   2.0 seems to be working reasonably well
MARGIN = 2.0

#Pick an Accuracy Margin
#   This value is used for training purposes
#   represents the proximity required between values to be associated with the same class
ACCURACY_MARGIN = 0.5

#Pick a K (for the K-Precision)
#   It is used to represent the k factor for calculating k-accuracy within the training process.
K = 12

#Pick a Batch Size
BATCH_SIZE = 16

#Pick a Backbone
#   The backbone represents the neural network within the siamese network, 
#   after which several linear layers will be applied to produce an embedding of size EMBEDDING_SIZE.
backbone = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)

#Pick a Learning Rate
lr = 1e-4

#Pick a number of Epochs
num_epochs = 500





# ====================================
#                CODE
# ====================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"You're using: {DEVICE}")
torch.set_default_dtype(torch.float32)
fix_random(42)
generator1 = torch.Generator().manual_seed(42)
workers = 0

#Images and Sketch
images_ds = ImageFolder(PHOTO_DATASET_PATH, transform = transforms.ToTensor())
images_loader = DataLoader(images_ds, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)
images_train_ds, images_val_ds = random_split(images_ds, (0.8, 0.2), generator = generator1)
images_train_dl = DataLoader(images_train_ds, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)
images_val_dl = DataLoader(images_val_ds, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)

sketches_ds = ImageFolder(SKETCHES_DATASET_PATH, transform = transforms.ToTensor())
sketches_loader = DataLoader(sketches_ds, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)
sketches_train_ds, sketches_val_ds, sketches_k_acc = random_split(sketches_ds, (0.8, 0.15, 0.05), generator = generator1)
small_sketches_loader = DataLoader(sketches_k_acc, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)

si_train_dataset = TRAIN_DATASET_TYPE(images_train_ds, sketches_train_ds, TRANSFORMATION)
si_train_loader = DataLoader(si_train_dataset, shuffle = True, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)
si_val_dataset = VAL_DATASET_TYPE(images_val_ds, sketches_val_ds)
si_val_loader = DataLoader(si_val_dataset, shuffle = False, num_workers = workers, pin_memory = True, batch_size = BATCH_SIZE)

criterion = CRITERION(MARGIN)
net = SiameseNetwork(output = OUTPUT_EMBEDDING, backbone = backbone).to(DEVICE)

optimizer = Adam(net.parameters(), lr = lr)
early_stopper = EarlyStopper(patience = 5, min_delta = 0)

history = training_loop(num_epochs, optimizer, net, 
                        si_train_loader, si_val_loader, small_sketches_loader, 
                        images_loader, criterion, K, DEVICE, ACCURACY_MARGIN,
                        early_stopping = early_stopper)

torch.save(net.state_dict(), WEIGHT_PATH)