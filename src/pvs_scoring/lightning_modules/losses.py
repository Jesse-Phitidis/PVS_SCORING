import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss, DeepSupervisionLoss
from monai.utils import DiceCEReduction, look_up_option, pytorch_after
from typing import Sequence, Callable


class MyCrossEntropyLoss(nn.CrossEntropyLoss):
    
    '''
    This class provides one improvement:
    
    1. The input weights do not need to be tensors already (useful for instantiating from yaml config).
    2. Weights are normalised.
    '''
    
    def __init__(self, weight=None, *args, **kwargs):
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight)
            weight = (weight / weight.sum()) * len(weight)
        super().__init__(weight=weight, *args, **kwargs)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=input.device)
        return F.cross_entropy(input, target, weight=weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)


class MyBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    '''
    This class provides one improvement:
    
    1. The input weights do not need to be tensors already.
    '''
    def __init__(self, pos_weight: Sequence | None = None, *args, **kwargs):
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        super().__init__(pos_weight=pos_weight, *args, **kwargs)


class MyDiceCELoss(DiceCELoss):
    
    '''
    This class provides three improvements:
    
    1. The input weights do not need to be tensors already.
    2. The option to normalise given weights in a sensible way.
    3. The option to force BCE to be used for multilabel settings. 
    '''
    
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        multilabel: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        dice_weight: Sequence | None = None,
        ce_weight: Sequence | None = None,
        sensible_weight_norm: bool = False,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` and ``weight`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            multilabel: use nn.BCEWithLogitsLoss for each channel to accomodate a multilabel setting.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            dice_weight: the weights for monai.losses.DiceCELoss. If include_background=False then it should not include
                a value for the first channel.
            ce_weight: the weight param for torch.nn.CrossEntropyLoss or the pos_weight param for torch.nn.BCEWithLogitsLoss 
                if the prediction is single channel or if multilabel=True.
            sensible_weight_norm: normalise given weights so they sum to the number of classes (after possibly removing 
                background for DiceLoss).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.multilabel = multilabel
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.sensible_weight_norm = sensible_weight_norm

        if dice_weight is not None:
            dice_weight = torch.tensor(dice_weight)
        if ce_weight is not None:
            ce_weight = torch.tensor(ce_weight)
        if dice_weight is not None and sensible_weight_norm:
            assert len(dice_weight) != 1, "Cannot use sensible_weight_norm for Dice on a single channel"
            dice_weight = (dice_weight / dice_weight.sum()) * len(dice_weight)
        if ce_weight is not None and sensible_weight_norm:
            assert len(ce_weight) != 1, "Cannot use sensible weight norm with BCE since background per channel is always given weight of 1"
            ce_weight = (ce_weight / ce_weight.sum()) * len(ce_weight)
        else:
            dice_weight = dice_weight
            ce_weight = ce_weight

        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=dice_weight,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(pos_weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)
        if input.shape[1] == 1 or self.multilabel:
            ce_loss = self.bce(input, target)
        else:
            ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss


class MyDeepSupervisionLoss(DeepSupervisionLoss):
    
    '''
    Add an argument to normalise the weights to sum to one like for nnU-Net.
    Add option to downsample the target with max pooling instead of interpolation.
    '''
    
    def __init__(self, norm: bool = False, max_pool: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.max_pool = max_pool
        
    def get_weights(self, *args, **kwargs) -> list[float]:
        weights = super().get_weights(*args, **kwargs)
        if not self.norm:
            return weights
        return (np.array(weights) / np.sum(weights)).tolist()

    def get_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape[2:] != target.shape[2:]:
            dims = len(input.shape[2:])
            if self.max_pool:
                max_pool = F.max_pool2d if dims == 2 else F.max_pool3d
                kernel = tuple(np.array(target.shape[2:]) // np.array(input.shape[2:]))
                target = max_pool(target, kernel)
            else:
                target = F.interpolate(target, size=input.shape[2:], mode=self.interp_mode)
        return self.loss(input, target)  # type: ignore[no-any-return]