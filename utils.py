from typing import Optional, Union
from collections import Iterable
from numbers import Real
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data import *
from model import *
from evaluation import *

# 对抗训练
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.25, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]` for one-vs-others mode (weight of negative class)
                        or :math:`\alpha_i \in \R`
                        vector of weights for each class (analogous to weight argument for CrossEntropyLoss)
        gamma (float): Focusing parameter :math:`\gamma >= 0`. When 0 is equal to CrossEntropyLoss
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’.
         ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
                in the output, uses geometric mean if alpha set to list of weights
         ‘sum’: the output will be summed. Default: ‘none’.
        ignore_index (Optional[int]): specifies indexes that are ignored during loss calculation
         (identical to PyTorch's CrossEntropyLoss 'ignore_index' parameter). Default: -100
    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> C = 5  # num_classes
        >>> N = 1 # num_examples
        >>> loss = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(N, C, requires_grad=True)
        >>> target = torch.empty(N, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[Union[float, Iterable]] = None,
        gamma: Real = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super(FocalLoss, self).__init__()
        if (
            alpha is not None
            and not isinstance(alpha, float)
            and not isinstance(alpha, Iterable)
        ):
            raise ValueError(
                f"alpha value should be None, float value or list of real values. Got: {type(alpha)}"
            )
        self.alpha: Optional[Union[float, torch.Tensor]] = (
            alpha
            if alpha is None or isinstance(alpha, float)
            else torch.FloatTensor(alpha)
        )
        if isinstance(alpha, float) and not 0.0 <= alpha <= 1.0:
            warnings.warn("[Focal Loss] alpha value is to high must be between [0, 1]")

        self.gamma: Real = gamma
        self.reduction: str = reduction
        self.ignore_index: int = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(input))
            )
        if input.shape[0] != target.shape[0]:
            raise ValueError(
                f"First dimension of inputs and targets should be same shape. "
                f"Got: {input.shape} and {target.shape}"
            )
        if len(input.shape) != 2 or len(target.shape) != 1:
            raise ValueError(
                f"input tensors should be of shape (N, C) and (N,). "
                f"Got: {input.shape} and {target.shape}"
            )
        if input.device != target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device
                )
            )

        # filter labels
        target = target.type(torch.long)
        input_mask = target != self.ignore_index
        target = target[input_mask]
        input = input[input_mask]
        # compute softmax over the classes axis
        pt = F.softmax(input, dim=1)
        logpt = F.log_softmax(input, dim=1)

        # compute focal loss
        pt = pt.gather(1, target.unsqueeze(-1)).squeeze()
        logpt = logpt.gather(1, target.unsqueeze(-1)).squeeze()
        focal_loss = -1 * (1 - pt) ** self.gamma * logpt

        weights = torch.ones_like(
            focal_loss, dtype=focal_loss.dtype, device=focal_loss.device
        )
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha = torch.tensor(self.alpha, device=input.device)
                weights = torch.where(target > 0, 1 - alpha, alpha)
            elif torch.is_tensor(self.alpha):
                alpha = self.alpha.to(input.device)
                weights = alpha.gather(0, target)

        tmp_loss = focal_loss * weights
        if self.reduction == "none":
            loss = tmp_loss
        elif self.reduction == "mean":
            loss = (
                tmp_loss.sum() / weights.sum()
                if torch.is_tensor(self.alpha)
                else torch.mean(tmp_loss)
            )
        elif self.reduction == "sum":
            loss = tmp_loss.sum()
        else:
            raise NotImplementedError(
                "Invalid reduction mode: {}".format(self.reduction)
            )
        return loss

def labels2file(p, outf_path):
    with open(outf_path, 'w+') as outf:
        for pi in p:
            outf.write(','.join([str(int(k)) for k in pi]) + '\n')

class MultilabelCELoss(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    def __init__(self):
        super(MultilabelCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return torch.mean(neg_loss + pos_loss)

def metric_cal(y_true, y_pred, task=1):
    y_true = y_true.cpu().numpy().tolist()
    y_pred = y_pred.cpu().numpy().tolist()
    if task == 1:
        y_pred = [p[0] for p in y_pred]
    assert len(y_true) == len(y_pred)
    if type(y_true) == int:
        y_true = [y_true]
    if task == 1:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1
