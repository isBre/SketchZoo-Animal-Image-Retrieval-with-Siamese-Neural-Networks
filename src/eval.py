from EmbeddingSpace import *
from torchvision import models
from Metrics import k_precision
from Utils import fix_random
from Networks import SiameseNetwork
from torchvision import transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageFolder

'''
SCRIPT DESCRIPTION:

This script is used to print a chart that compares various k-precision values of different models using a bar graph.
'''

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"We're using {DEVICE}")
fix_random(42)
generator1 = torch.Generator().manual_seed(42)
workers = 0

dataset_paths = {'mini' : ["../Mini Dataset/photo", "../Mini Dataset/sketch"], 
                 'full' : ["../Full Dataset/256x256/photo", "../Full Dataset/256x256/sketch"]}





# ====================================
#               CONFIG
# ====================================

#Pick a Dataset (you can use the dictionary up here as reference)
DATASET_NAME = 'full'
PHOTO_DATASET_PATH, SKETCHES_DATASET_PATH = dataset_paths[DATASET_NAME]

#Pick a K (for the K-Precision)
#   It is used to represent the k factor for calculating k-accuracy 
K = 12

#Pick a Batch Size
BATCH_SIZE = 16

#Create the dictionary for the final graph
#   For each experiment pick the backbone, the embedding_size and the weight_path
dict={0: {'backbone' : models.resnet18(), 'embedding_size' : 16, 'weight_path' : '../weights/full-16-contrastive.pth'},
      1: {'backbone' : models.resnet34(), 'embedding_size' : 16, 'weight_path' : '../weights/full-16-contrastive-resnet34.pth'},
      2: {'backbone' : models.resnet50(), 'embedding_size' : 16, 'weight_path' : '../weights/full-16-contrastive-resnet50.pth'},
      3: {'backbone' : models.resnet101(), 'embedding_size' : 16, 'weight_path' : '../weights/full-16-contrastive-resnet101.pth'}}





# ====================================
#                CODE
# ====================================

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


results = []

for k in list(dict.keys()):
    net = SiameseNetwork(output = dict[k]['embedding_size'], backbone = dict[k]['backbone']).to(DEVICE)
    net.load_state_dict(torch.load(dict[k]['weight_path']))
    net = net.eval()
    with torch.no_grad():
        embedding_space1 = EmbeddingSpace(net, images_val_dl, DEVICE)
        results.append(k_precision(net, small_sketches_loader, embedding_space1, K, DEVICE))
        print(f'Model {k} has K@{K} = {results[k]:.2f}')


plt.bar(["Resnet18", "Resnet34", "Resnet50", "resnet101"], results)
plt.show()


    
    



