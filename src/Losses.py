import torch
from torch.nn import Module
from torch.nn import Module
import torch.nn.functional as F


class ContrastiveLoss(Module):

    def __init__(self, margin: int) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs : torch.Tensor, label : torch.Tensor):
        if outputs.size()[0] == 2:
            euclidean_distance = F.pairwise_distance(outputs[0], outputs[1], keepdim = True)
            # Calculate the squared euclidean distance for non-matching pairs
            non_matching_loss = (1 - label) * torch.pow(euclidean_distance, 2)
            # Calculate the margin loss for matching pairs, clamping the distance to be non-negative
            matching_loss = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
            # Compute the average loss (contrastive loss)
            loss_contrastive = torch.mean(non_matching_loss + matching_loss)
            return loss_contrastive
        raise ValueError('This is the Contrastive Loss but more (or less) than 2 tensors were unpacked')

class TripletLoss(Module):

    def __init__(self, margin: int) -> None:
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if len(outputs) == 3:
            positive_distance = F.pairwise_distance(outputs[0], outputs[1])
            negative_distance = F.pairwise_distance(outputs[0], outputs[2])
            losses = torch.relu(positive_distance - negative_distance + self.margin)
            return torch.mean(losses)
        raise ValueError('This is the Triplet Loss but more (or less) than 3 tensors were unpacked')