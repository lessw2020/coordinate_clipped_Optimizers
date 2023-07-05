
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
import numpy as np
from typing import List

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
                        decoupled_wd=use_decoupled_weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(CoordinateWiseClipping_AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('use_decoupled_weight_decay', True)

    
    @torch.no_grad()
    def restart_optimizer(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults['top_k_percent'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_coord_threshold = torch.zeros(1, device=device)

            top_k_percentage = torch.tensor(self.defaults['top_k_percent'],
                                         device=device)
            raw_grad_sizes = []
            
            # bool where True = +
            raw_grad_sign = []
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad:
                        grad = p.grad
                        raw_grad_sign.append(grad > 0)
                        raw_grad_sizes.append(torch.abs(grad))
                   
            global_coord_max = np.percentile(raw_grad_sizes, global_coord_threshold, method = "nearest")
            
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            
            beta1, beta2 = group['betas']
            
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1**group['step']
            bias_correction2 = 1.0 - beta2**group['step']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                
                beta1=beta1,
                beta2=beta2,
               
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                decouple_wd=group['decouple_wd'],
                global_coord_max=global_coord_max,
                raw_grad_sizes = raw_grad_sizes, 
                raw_grad_sign = raw_grad_signs,
                
            )

            
        _single_tensor_cwcAdamW(**kwargs)

        return loss


def _single_tensor_cwcAdamW(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    bias_correction1: float,
    bias_correction2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    decouple_wd=group['decouple_wd'],
    global_coord_max,
    raw_grad_sizes,
    raw_grad_sign,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        k_grad = grad
        
        torch.clamp(k_grad, min = -global_coord_max, max = global_coord_max)
        

        exp_avg.mul_(beta1).add_(k_grad, alpha=1 - beta1)  # momentum_term
        
        exp_avg_sq.mul_(beta2).add_(grad, alpha=1-beta2)  # variance_term

        denom = (exp_avg_sq.sqrt().add_(eps))
        step_size_diff = lr * beta2 / bias_correction2
        step_size = lr / bias_correction1

        if not decouple_wd:
            param.mul_(1 - lr * weight_decay)
            param.addcdiv_(exp_avg, denom, value=-step_size)
            
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)
            
            param.div_(1 + lr * weight_decay)

