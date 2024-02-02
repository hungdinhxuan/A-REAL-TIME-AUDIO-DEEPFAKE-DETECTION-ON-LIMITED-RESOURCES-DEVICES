import torch
import numpy as np
import os
from torch import Tensor
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@torch.jit.script
def pad(x, max_len: int = 64600) -> Tensor:
    x_len = torch.tensor(x.shape[0])
    max_len = torch.tensor(max_len)

    if torch.ge(x_len, max_len).item():
        return x[:max_len]
        # need to pad
    num_repeats = int((max_len / x_len).ceil().item())
        
    padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
    return padded_x

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, model_save_path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_save_path = model_save_path
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info('EarlyStopping counter: %s out of %s - Current best score: %s', self.counter, self.patience, self.best_score)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.model_save_path, 'best_checkpoint_{}.pth'.format(epoch)))
        # Remove previous best model to save memory
        if epoch > 0:
            previous_best_model_path = os.path.join(self.model_save_path, 'best_checkpoint_{}.pth'.format(epoch-1))
            if os.path.exists(previous_best_model_path):
                os.remove(previous_best_model_path)
                logger.debug(f'Removed previous best model at {previous_best_model_path}')

        self.val_loss_min = val_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count