# SketchZoo-Animal-Image-Retrieval-with-Siamese-Neural-Networks
SketchZoo is a state-of-the-art project that utilizes Siamese neural networks for efficient animal image retrieval. With the power of contrastive and triplet loss functions, SketchZoo accurately matches hand-drawn animal sketches to corresponding images. It also provides a user-friendly GUI interface for convenient image retrieval based on sketches

## Dataset Informations
The dataset used is called the Sketchy dataset and can be downloaded at the following link: [https://sketchy.eye.gatech.edu/](https://sketchy.eye.gatech.edu/). This dataset consists of a hundred classes. To simplify the problem, I chose to focus only on the classes associated with animals, resulting in a selection of **55 animal classes**.

![](/imgs/Lion.gif)

# Files
- **imgs**: This directory contains all the images used in the presentation.
- **src**: This directory contains all the `.py` files.
- **weights**: This directory contains the best weights obtained. The weight file naming format is `{dataset-name}-{embedding-size}-{loss-used}-[OPTIONAL: model-used].pth`.

## src Files
- **Dataset.py**: This file contains the main dataset classes, including `ContrastiveDataset`, `TripletDataset`, `AugmentedContrastiveDataset`, and `AugmentedTripletDataset`. These classes are used for utilizing either Contrastive Loss or Triplet Loss.
- **EarlyStopper.py**: This file defines the EarlyStopper class.
- **EmbeddingSpace.py**: This file defines the EmbeddingSpace class, along with the associated functions for retrieving elements from it.
- **Losses.py**: This file contains the definitions for ContrastiveLoss and TripletLoss.
- **Network.py**: This file contains the implementation of the main network, `SiameseNetwork`.
- **TrainingFunctions.py**: This file includes various training functions such as `trainingloop`, `train`, `validate`, and `get_correct`.
- **Utils.py**: This file provides utility functions, including `fix_random`, used to ensure reproducibility.
- **canvas.py**: This executable script launches the GUI program. Make sure to configure the correct file in the program settings.
- **eval.py**: This script generates a graph comparing different P@K values for various models. Ensure to set the correct file in the configuration.
- **space.py**: This script produces a scatter plot of a model with an embedding size of 2. Remember to set the correct file in the configuration.
- **train.py**: This script is used for training the model. Adjust the configuration settings with the appropriate file.

![](/imgs/Snail.gif)

# How to Use
First and foremost, a dataset is required that can be useful for training a model from scratch or utilizing pre-trained weights to create the embedding space. It is also important that this dataset is in the [ImageFolder format](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html). Once this is done, you can proceed to one of the listed scripts, configure the initial parameters, and begin training or initiate the GUI.

![](/imgs/Shark.gif)




