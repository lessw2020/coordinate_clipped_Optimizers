
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
        betas (Tuple[float, float], optional): coefficients used for
            first- and second-order moments. (default: (0.90, 0.999)
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay
            (L2 penalty) (default: 0)
        percent_to_clip (float, optional): top k-percent coordinate-wise clipping (default: 0.50)
        use_decoupled_weight_decay (bool): how to perform the decoupled weight decay
            (default: True)
        
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.90, 0.99),
                 eps=1e-8,
                 weight_decay=0.0,
                 percent_to_clip: float = 0.50,
                 use_decoupled_weight_decay: bool = True):
        if not 0.0 <= max_grad_norm:
            raise ValueError('Invalid Max grad norm: {}'.format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        if not 0.0 <= percent_to_clip < 1.0:
            raise ValueError('Invalid percent_to_clip parameter: {}'.format(
                percent_to_clip))
            
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        top_k_percent=percent_to_clip,
                        no_prox=no_prox,
                        decoupled_wd=use_decoupled_weight_decay)
        super().__init__(params, defaults)
        
        

