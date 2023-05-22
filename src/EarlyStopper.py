import numpy as np

class EarlyStopper:
  """
  Represent an object that is able to tell when the train need to stop.
  Needs to be put inside the train and keep updated, when early_stop returns
  true we need to stop the train
  """
  def __init__(self, patience : int = 1, min_delta : int = 0):
      self.patience = patience
      self.min_delta = min_delta
      self.counter = 0
      self.min_validation_loss = np.inf

  def early_stop(self, validation_loss : float) -> bool:
      if validation_loss < self.min_validation_loss:
          self.min_validation_loss = validation_loss
          self.counter = 0
      elif validation_loss > (self.min_validation_loss + self.min_delta):
          self.counter += 1
          if self.counter >= self.patience:
              return True
      return False