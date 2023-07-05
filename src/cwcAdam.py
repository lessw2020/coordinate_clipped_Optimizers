
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

class CoordinateWiseClipping_AdamW(Optimizer):
    """
    Implements a pytorch version of coordinate-wise clipped AdamW
    Coordinate-wise clipping was proposed in:
    https://arxiv.org/abs/2306.00204  
    "Toward Understanding Why Adam Converges Faster Than SGD for Transformers"
    
    Arguments:
        params (iterable): iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for
            first- and second-order moments. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay
            (L2 penalty) (default: 0)
        percent_to_clip (float, optional): top k-percent coordinate-wise clipping (default: 0.50)
        use_decoupled_weight_decay (bool): how to perform the decoupled weight decay
            (default: True)
        
    """
