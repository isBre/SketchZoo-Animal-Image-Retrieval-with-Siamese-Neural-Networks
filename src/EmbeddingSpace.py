import torch
from torch.nn import Module
from torch.utils.data import DataLoader


class EmbeddingSpace():

  '''
  Every image should be represented as an embedding vector,
  Here we have a representation of an embedding space
  '''

  def __init__(self,  model: Module,
                      loader_images : DataLoader,
                      device: torch.device):
    
    """Create the embedding space

    Args:
        model: the CNN backbone.
        loader_images: dataloader of images.
    """
    self.device = device
    self.model = model.eval().to(self.device)
    self.embeddings = torch.tensor([]).to(self.device)
    self.classes = torch.tensor([]).to(self.device)

    for idx_batch, (images, images_class) in enumerate(loader_images):

      images, images_class = images.to(self.device), images_class.to(self.device)
      with torch.no_grad():
        out = torch.squeeze(model.forward_once(images)).to(self.device)
        self.embeddings = torch.cat((self.embeddings, out))
        self.classes = torch.cat((self.classes, images_class))





  def top_k(self, sketches: torch.Tensor, k: int):
    
    with torch.no_grad():
      sketch_embeddings = torch.squeeze(self.model.forward_once(sketches.to(self.device)))[None,:]

    distances = torch.cdist(self.embeddings, sketch_embeddings)
    topk_distances, topk_indices = torch.topk(distances, k, largest = False, dim = 0)

    return topk_distances, topk_indices




  def top_k_batch(self, sketches: torch.Tensor, k : int):
    
    with torch.no_grad():
      sketch_embeddings = torch.squeeze(self.model.forward_once(sketches.to(self.device)))

    distances = torch.cdist(self.embeddings, sketch_embeddings)
    topk_distances, topk_indices = torch.topk(distances, k, largest = False, dim = 0)

    return torch.permute(topk_distances, (1,0)), torch.permute(topk_indices, (1,0))
  
