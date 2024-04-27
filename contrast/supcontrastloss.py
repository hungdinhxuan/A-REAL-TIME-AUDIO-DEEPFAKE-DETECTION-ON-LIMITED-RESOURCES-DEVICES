"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
import torch.nn.functional as torch_nn_func
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

#####################
# supervised contrastive loss [2]
#####################


def sim_metric_seq(mat1, mat2):
    return torch.bmm(mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)


def supcon_loss(input_feat,
                labels=None, mask=None, sim_metric=sim_metric_seq,
                t=0.07, contra_mode='all', length_norm=False):
    """
    loss = SupConLoss(feat, 
                      labels = None, mask = None, sim_metric = None, 
                      t=0.07, contra_mode='all')
    input
    -----
      feat: tensor, feature vectors z [bsz, n_views, ...].
      labels: ground truth of shape [bsz].
      mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
      sim_metric: func, function to measure the similarity between two 
            feature vectors
      t: float, temperature
      contra_mode: str, default 'all'
         'all': use all data in class i as anchors
         'one': use 1st data in class i as anchors
      length_norm: bool, default False
          if True, l2 normalize feat along the last dimension

    output
    ------
      A loss scalar.

    Based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    Example:
      feature = torch.rand([16, 2, 1000], dtype=torch.float32)
      feature = torch_nn_func.normalize(feature, dim=-1)
      label = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 1, 1, 1, 1, 1], 
               dtype=torch.long)
      loss = supcon_loss(feature, labels=label)
    """
    if length_norm:
        feat = torch_nn_func.normalize(input_feat, dim=-1)
    else:
        feat = input_feat

    if len(feat.shape) < 3:
        raise ValueError('`feat` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(feat.shape) > 3:
        feat = feat.view(feat.shape[0], feat.shape[1], -1)

    # batch size
    bs = feat.shape[0]
    # device
    dc = feat.device
    # dtype
    dt = feat.dtype
    # number of view
    nv = feat.shape[1]

    # get the mask
    # mask[i][:] indicates the data that has the same class label as data i
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(bs, dtype=dt, device=dc)
    elif labels is not None:
        labels = labels.view(-1, 1)

        if labels.shape[0] != bs:

            raise ValueError(
                f'Num of labels does not match num of features, labels.shape[0] = {labels.shape[0]}, bs = {bs}')
        mask = torch.eq(labels, labels.T).type(dt).to(dc)
    else:
        mask = mask.type(dt).to(dc)

    # prepare feature matrix
    # -> (num_view * batch, feature_dim, ...)
    contrast_feature = torch.cat(torch.unbind(feat, dim=1), dim=0)
    #
    if contra_mode == 'one':
        # (batch, feat_dim, ...)
        anchor_feature = feat[:, 0]
        anchor_count = 1
    elif contra_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = nv
    else:
        raise ValueError('Unknown mode: {}'.format(contra_mode))

    # compute logits
    # logits_mat is a matrix of size [num_view * batch, num_view * batch]
    # or [batch, num_view * batch]

    if sim_metric is not None:
        logits_mat = torch.div(
            sim_metric(anchor_feature, contrast_feature), t)
    else:
        logits_mat = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), t)

    # print(anchor_feature.shape)
    # mask based on the label
    # -> same shape as logits_mat
    mask_ = mask.repeat(anchor_count, nv)
    # mask on each data itself (
    self_mask = torch.scatter(
        torch.ones_like(mask_), 1,
        torch.arange(bs * anchor_count).view(-1, 1).to(dc),
        0)
    # print(self_mask)
    #
    mask_ = mask_ * self_mask
    # print(mask_)

    # for numerical stability, remove the max from logits
    # see https://en.wikipedia.org/wiki/LogSumExp trick
    # for numerical stability
    logits_max, _ = torch.max(logits_mat * self_mask, dim=1, keepdim=True)
    logits_mat_ = logits_mat - logits_max.detach()
    # compute log_prob
    exp_logits = torch.exp(logits_mat_ * self_mask) * self_mask
    log_prob = logits_mat_ - torch.log(exp_logits.sum(1, keepdim=True))

    # print("log_prob.shape", log_prob.shape)
    # compute mean of log-likelihood over positive
    # print(mask_ * log_prob)
    mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)
    # print("mean_log_prob_pos.shape", mean_log_prob_pos.shape)
    # print(mean_log_prob_pos)
    # loss
    loss = - mean_log_prob_pos
    loss = loss.view(anchor_count, bs).mean()

    return loss
