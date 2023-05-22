import torch
from torch.nn import Module, Sequential, Linear, ReLU, Conv2d, MaxPool2d, AdaptiveMaxPool2d, Dropout
from torchvision import models

'''
The input received here is
(BATCH_SIZE, INPUTS_LEN, CHANNELS = 3, HEIGHT = 256, WIDTH = 256)
'''

class SiameseNetwork(Module):

    def __init__(self, output : int, backbone: models):
        super(SiameseNetwork, self).__init__()

        self.backbone = backbone
        
        self.backbone.fc = Sequential (
            Linear(self.backbone.fc.in_features, 256),
            ReLU(inplace = True),
            Linear(256, 128),
            ReLU(inplace = True),
            Linear(128, output)
        )


    def forward_once(self, x : torch.Tensor):
        return self.backbone(x)


    def forward(self, inputs : torch.Tensor):

        if len(inputs) == 2:
            return torch.stack((self.forward_once(inputs[0]),
                                self.forward_once(inputs[1])))
        
        if len(inputs) == 3:
            return torch.stack((self.forward_once(inputs[0]),
                                self.forward_once(inputs[1]),
                                self.forward_once(inputs[2]),))

        raise ValueError(f'This is the SiameseBaseline, the number of input tensor is not 2 or 3 {inputs.size()}')