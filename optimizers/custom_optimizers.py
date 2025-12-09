"""
Custom optimizer implementations for comparison
Implements SGD, Momentum, Adagrad, and Adam from scratch
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    
    def __init__(self, params, lr=0.01, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Update parameters
                p.add_(grad, alpha=-lr)
        
        return loss


class MomentumSGD(Optimizer):
    """SGD with Momentum"""
    
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # Initialize velocity
                if len(param_state) == 0:
                    param_state['velocity'] = torch.zeros_like(p)
                
                v = param_state['velocity']
                
                # Update velocity: v = momentum * v + grad
                v.mul_(momentum).add_(grad)
                
                # Update parameters: theta = theta - lr * v
                p.add_(v, alpha=-lr)
        
        return loss


class Adagrad(Optimizer):
    """Adagrad optimizer"""
    
    def __init__(self, params, lr=0.01, eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # Initialize accumulated squared gradients
                if len(param_state) == 0:
                    param_state['sum_sq_grad'] = torch.zeros_like(p)
                
                sum_sq_grad = param_state['sum_sq_grad']
                
                # Accumulate squared gradients
                sum_sq_grad.add_(grad * grad)
                
                # Compute adaptive learning rate
                std = sum_sq_grad.sqrt().add_(eps)
                
                # Update parameters
                p.addcdiv_(grad, std, value=-lr)
        
        return loss


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # Initialize first and second moment estimates
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['m'] = torch.zeros_like(p)
                    param_state['v'] = torch.zeros_like(p)
                
                m = param_state['m']
                v = param_state['v']
                param_state['step'] += 1
                step = param_state['step']
                
                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected moment estimates
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                
                # Update parameters
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        
        return loss


class RAdam(Optimizer):
    """Rectified Adam optimizer"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                m, v = state['m'], state['v']
                state['step'] += 1
                step = state['step']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute variance rectification term
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2

                if rho_t > 4:
                    rect = ((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    rect = rect.sqrt()
                    denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr * rect / bias_correction1
                    p.addcdiv_(m, denom, value=-step_size)
                else:
                    # Unrectified update (like SGD with momentum)
                    p.add_(m, alpha=-lr / bias_correction1)

        return loss
