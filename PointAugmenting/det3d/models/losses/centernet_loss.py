import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 

    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  Supports optional class-wise weighting for positive/negative terms.
  '''
  def __init__(self, alpha: float = 2.0, beta: float = 4.0):
    super(FastFocalLoss, self).__init__()
    self.alpha = alpha  # exponent for (1 - p) in positive term
    self.beta = beta    # exponent for (1 - target) in negative modulator

  def forward(self, out, target, ind, mask, cat, class_weights=None, alpha: float = None, beta: float = None):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
      class_weights: Optional list/1D tensor of length C for class-wise weights
      alpha, beta: Optional override of exponents for this call
    '''
    mask = mask.float()
    a = self.alpha if alpha is None else alpha
    b = self.beta if beta is None else beta

    # Negative loss over the entire heatmap
    gt = torch.pow(1 - target, b)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt

    # Apply per-class weights to negative term if provided
    if class_weights is not None:
      if not torch.is_tensor(class_weights):
        class_weights = torch.tensor(class_weights, dtype=out.dtype, device=out.device)
      else:
        class_weights = class_weights.to(device=out.device, dtype=out.dtype)
      w = class_weights.view(1, -1, 1, 1)  # 1 x C x 1 x 1
      neg_loss = neg_loss * w
    neg_loss = neg_loss.sum()

    # Positive loss at annotated peaks
    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_term = torch.log(pos_pred) * torch.pow(1 - pos_pred, a)

    if class_weights is not None:
      # Gather class weights for each positive according to its category id
      if not torch.is_tensor(class_weights):
        class_weights = torch.tensor(class_weights, dtype=out.dtype, device=out.device)
      w_pos = class_weights.gather(0, cat.view(-1)).view_as(cat).to(out.dtype)  # B x M
      pos_term = pos_term * w_pos.unsqueeze(2)

    pos_loss = pos_term * mask.unsqueeze(2)
    pos_loss = pos_loss.sum()

    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos
