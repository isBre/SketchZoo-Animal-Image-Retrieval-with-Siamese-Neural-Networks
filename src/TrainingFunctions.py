import torch
from torch import optim, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.modules import loss
from torch import device
import torch.nn.functional as F

from timeit import default_timer as timer
from typing import Dict, Callable, Tuple

from EmbeddingSpace import EmbeddingSpace
from EarlyStopper import EarlyStopper
from Metrics import k_precision





def training_loop(num_epochs: int, 
                  optimizer: optim,
                  model: Module, 
                  si_train_loader: DataLoader, 
                  si_val_loader: DataLoader,
                  sketches_test_loader: DataLoader,
                  images_dataloader: DataLoader,
                  loss_func: loss,
                  k: int,
                  device: device,
                  accuracy_margin: float,
                  verbose: bool = True,
                  early_stopping: EarlyStopper = None) -> Dict:

    # Start the timer in order to obtain the time needed to entirely train the model
    loop_start = timer()

    train_losses_values = []
    val_losses_values = []
    train_acc_values = []
    val_acc_values = []
    k_prec_values = []

    # For every epoch
    for epoch in range(1, num_epochs + 1):

      # Start the timer in order to obtain the time needed to train in this epoch
      time_start = timer()

      # Obtain Loss and Accuracy for the train step
      loss_train, accuracy_train = train(optimizer, model, si_train_loader, 
                                         loss_func, accuracy_margin, device)

      # Obtain Loss and Accuracy for the validate step
      loss_val, accuracy_val = validate(model, si_val_loader, loss_func, accuracy_margin, device)
      
      # Calculate the k-accuracy
      embedding_space = EmbeddingSpace(model, images_dataloader, device)
      k_prec = k_precision(model, sketches_test_loader, embedding_space, k, device)   

      #Stop the timer for this step
      time_end = timer()

      # Update history
      train_losses_values.append(loss_train)
      train_acc_values.append(accuracy_train)
      val_losses_values.append(loss_val)
      val_acc_values.append(accuracy_val)
      k_prec_values.append(k_prec)
      
      # Metrics Print
      lr =  optimizer.param_groups[0]['lr']
      if verbose:            
          print(f'Epoch: {epoch} '
                f' Lr: {lr:.8f} '
                f' Loss: Train = [{loss_train:.4f}] - Val [{loss_val:.4f}]'
                f' Accuracy: Train = [{accuracy_train:.2f}%] - Val = [{accuracy_val:.2f}%] '
                f' P@{k}: [{k_prec:.2f}%]'
                f' Time one epoch (s): {(time_end - time_start):.2f} ')
    
      if early_stopping is not None:
        if early_stopping.early_stop(loss_val):
          print(f'--- Early Stopping ---')     
          break

    # Stop the timer for the entire training
    loop_end = timer()

    # Calculate total time
    time_loop = loop_end - loop_start

    # Metrics Print
    if verbose:
        print(f'Time for {epoch-1} epochs (s): {(time_loop):.3f}') 
        
    return {'train_loss_values': train_losses_values,
            'train_acc_values': train_acc_values,
            'val_loss_values' : val_losses_values,
            'val_acc_values': val_acc_values,
            'k_acc_values' : k_prec,
            'time': time_loop}





def train(optimizer: optim,
          model: Module,
          dataloader: DataLoader,
          loss_func: Callable[[Tensor, Tensor], float],
          accuracy_margin: float,
          device: device) -> Tuple[float, float]:


    # Initialize Metrics
    correct = 0.0
    samples_train = 0
    loss_train = 0
    num_batches = len(dataloader)

    # IMPORTANT: from now on, since we will introduce batch norm, 
    # we have to tell PyTorch if we are training or evaluating our model
    model.train()

    # Loop inside the train_loader
    # The batch size is definited inside the train_loader
    for inputs, target in dataloader:

      # In order to speed up the process I want to use the current device
      inputs, target = inputs.to(device), target.to(device)

      inputs = torch.permute(inputs, (1, 0, 2, 3, 4))

      # Set the gradient of the available parameters to zero
      optimizer.zero_grad()

      # Get the output of the model
      outputs = model(inputs)
    
      # Here the model calculate the loss comparing true values and obtained values
      loss = loss_func(outputs, target)

      # Update the total loss adding the loss of this particular batch
      loss_train += loss.item() * len(inputs[0])

      # Update the number of analyzed images
      samples_train += len(inputs[0])
      
      # Compute the gradient
      loss.backward()

      # Update parameters considering the loss.backward() values
      optimizer.step()

      # Update the number of correct predicted values adding the correct value of this batch
      correct += get_correct(outputs, target, accuracy_margin, device)


    loss_train /= samples_train
    accuracy_training = 100. * correct / samples_train
    return loss_train, accuracy_training





def validate(model: Module,
             dataloader: DataLoader,
             loss_func: Callable[[Tensor, Tensor], float],
             accuracy_margin: float,
             device: torch.device) -> Tuple[float, float]:

  # Corrected Labeled samples
  correct = 0

  # Images in the batch
  samples_val = 0

  # Loss of the Valuation Set
  loss_val = 0.

  # IMPORTANT: from now on, since we will introduce batch norm, we have to tell PyTorch if we are training or evaluating our model
  model = model.eval()

  # Context-manager that disabled gradient calculation
  with torch.no_grad():

    # Loop inside the data_loader
    # The batch size is definited inside the data_loader
    for inputs, target in dataloader:

      # In order to speed up the process I want to use the current device
      inputs, target = inputs.to(device), target.to(device)
      inputs = torch.permute(inputs, (1, 0, 2, 3, 4))

      # Get the output of the model
      # I need to squeeze because of the dimension of the output (x, 1), I want just (x)
      outputs = model(inputs)

      # Here the model calculate the loss comparing true values and obtained values
      # Here i need to cast to float32 because: labels is long and outputs is float32
      loss = loss_func(outputs, target)

      # Update metrics
      loss_val += loss.item() * len(inputs[0])
      samples_val += len(inputs[0])
      correct += get_correct(outputs, target, accuracy_margin, device)

  loss_val /= samples_val
  accuracy = 100. * correct / samples_val
  return loss_val, accuracy





def get_correct(outputs: torch.Tensor,
                labels: torch.Tensor,
                accuracy_margin: float,
                device = device) -> float:

    if len(outputs) == 2:

      distances = F.pairwise_distance(outputs[0], outputs[1]).to(device)
      results = torch.tensor([0 if d < accuracy_margin else 1 for d in distances]).to(device)
      matches = torch.eq(results, torch.squeeze(labels)).float().to(device)

      return torch.sum(matches)


    if len(outputs) == 3:

      positive_distances = F.pairwise_distance(outputs[0], outputs[1]).to(device)
      negative_distances = F.pairwise_distance(outputs[0], outputs[2]).to(device)
      results = torch.tensor([1 if p < n else 0 for p, n in zip(positive_distances, negative_distances)]).to(device)

      return torch.sum(results)
    
    raise ValueError(f'Output with wrong dimension in get_correct{outputs.size()}')