import torch
import numpy as np
import os
from torch import Tensor
import logging
import subprocess
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
            logger.info('EarlyStopping counter: %s out of %s - Current best score: %s',
                        self.counter, self.patience, self.best_score)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        path_save = os.path.join(
            self.model_save_path, 'best_checkpoint_{}.pth'.format(epoch))
        torch.save(model.state_dict(), path_save)

        command = f"""
        CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "{path_save}" --eval_output="./{path_save}.txt" --batch_size_eval=300 --wrapper_ssl --dataset='cnsl' --database_path='/home/hungdx/Datasets/supcon_cnsl_feb07' --protocols_path='protocol.txt' --student_model_type='Distil_W2V2BASE_Linear'
        """

        # subprocess.Popen(command, shell=True)
        with open(os.devnull, 'w') as devnull:
            subprocess.Popen(command, shell=True, stdin=devnull,
                             stdout=devnull, stderr=devnull)
        # Remove previous best model to save memory
        # if epoch > 0:
        #     previous_best_model_path = os.path.join(self.model_save_path, 'best_checkpoint_{}.pth'.format(epoch-1))
        #     if os.path.exists(previous_best_model_path):
        #         os.remove(previous_best_model_path)
        #         logger.debug(f'Removed previous best model at {previous_best_model_path}')

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


def get_params(model, ignore_auxiliary_head=True):
    if not ignore_auxiliary_head:
        params = sum([m.numel() for m in model.parameters()])
    else:
        params = sum(
            [m.numel() for k, m in model.named_parameters() if 'auxiliary_head' not in k])
    return params


def get_flops(model, input_shape=(1, 64600)):
    if hasattr(model, 'flops'):
        return model.flops(input_shape)
    else:
        return get_flops_hook(model, input_shape)


def get_flops_hook(model, input_shape=(1, 64600)):
    is_training = model.training
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        assert self.in_channels % self.groups == 0

        kernel_ops = self.kernel_size[0] * self.kernel_size[
            1] * (self.in_channels // self.groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement()

        flops = batch_size * weight_ops
        list_linear.append(flops)

    def foo(net, hook_handle):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                hook_handle.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handle.append(net.register_forward_hook(linear_hook))
            return
        for c in childrens:
            foo(c, hook_handle)

    hook_handle = []
    foo(model, hook_handle)
    input = torch.rand(
        *input_shape).unsqueeze(0).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        out = model(input)
    for handle in hook_handle:
        handle.remove()

    total_flops = sum(sum(i) for i in [list_conv, list_linear])
    model.train(is_training)
    return total_flops
