import matplotlib.pyplot as plt
from EmbeddingSpace import *
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from Networks import SiameseNetwork

'''
SCRIPT DESCRIPTION:

This script is responsible for plotting the embedding space of a model in a 2D space (where EMBEDDING_SIZE = 2).
'''

dataset_paths = {'mini' : ["../Mini Dataset/photo", "../Mini Dataset/sketch"], 
                 'full' : ["../Full Dataset/256x256/photo", "../Full Dataset/256x256/sketch"]}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# ====================================
#               CONFIG
# ====================================

#Pick a Dataset (you can use the dictionary up here as reference)
DATASET_NAME = 'mini'
PHOTO_DATASET_PATH, SKETCHES_DATASET_PATH = dataset_paths[DATASET_NAME]

#Pick an embedding size
#   MUST BE EQUAL TO 2
OUTPUT_EMBEDDING = 2

#Pick a Backbone
#   The backbone represents the neural network within the siamese network, 
#   after which several linear layers will be applied to produce an embedding of size EMBEDDING_SIZE.
backbone = models.resnet18()

#Load a model (the embedding_size MUST be equal to 2)
net = SiameseNetwork(output = OUTPUT_EMBEDDING, backbone = backbone).to(DEVICE)
net.load_state_dict(torch.load("../weights/mini-2-contrastive.pth"))

#Pick a Batch Size
BATCH_SIZE = 16

images_ds = ImageFolder(PHOTO_DATASET_PATH, transform = transforms.ToTensor())
images_loader = DataLoader(images_ds, shuffle = False, batch_size = BATCH_SIZE)
images_train_ds, images_val_ds = random_split(images_ds, (0.8, 0.2))
images_train_dl = DataLoader(images_train_ds, shuffle = False, batch_size = BATCH_SIZE)
images_val_dl = DataLoader(images_val_ds, shuffle = False, batch_size = BATCH_SIZE)

#Pick the dataset that you want to embed
loader = images_loader

embedding_space = EmbeddingSpace(net, loader, DEVICE)

x = torch.permute(embedding_space.embeddings, (1,0)).tolist()[0]
y = torch.permute(embedding_space.embeddings, (1,0)).tolist()[1]

colours_dict = {0 : 'brown', 1 : 'green', 2 : 'blue', 3: 'yellow', 4 :'black'}
colors = [colours_dict[c.item()] for c in embedding_space.classes]

plt.scatter(x, y, color = colors)
plt.show()