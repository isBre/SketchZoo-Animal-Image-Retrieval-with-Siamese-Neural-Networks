import torch
from torch.nn import Module
from torch.nn import Module
import torch.nn.functional as F

class ContrastiveLoss(Module):

    def __init__(self, margin : int):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs : torch.Tensor, label : torch.Tensor):

        if outputs.size()[0] == 2:
      
            euclidean_distance = F.pairwise_distance(outputs[0], outputs[1], keepdim = True)
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min = 0.0), 2))
                
            return loss_contrastive
      
        raise ValueError('This is the Contrastive Loss but more (or less) than 2 tensors were unpacked')





class TripletLoss(Module):
    def __init__(self, margin : int):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, outputs : torch.Tensor, labels : torch.Tensor) -> torch.Tensor:
        
        if len(outputs) == 3:

            positive_distance = F.pairwise_distance(outputs[0], outputs[1])
            negative_distance = F.pairwise_distance(outputs[0], outputs[2])
            losses = torch.relu(positive_distance - negative_distance + self.margin)

            return torch.mean(losses)
        
        raise ValueError('This is the Triplet Loss but more (or less) than 3 tensors were unpacked')