
import torch
from torch.optim import Optimizer
class StochasticArmijoSGD(Optimizer):
    def __init__(self, params, initial_step=1.0, c=0.001, tau=0.7, max_backtracks=20):
        """
        Stochastic Armijo line search optimizer. A vanilla SGD with armijo line search.

        Args:
            params: model parameters
            initial_step: initial step size (alpha_0)
            c: Armijo condition parameter 0-1
            tau: step size reduction factor 0 -1
            max_backtracks: max number of backtracking steps
        """
        defaults = dict(initial_step=initial_step, c=c, tau=tau, max_backtracks=max_backtracks)
        super(StochasticArmijoSGD, self).__init__(params, defaults)

    def step(self, closure):
        if closure is None:
            raise ValueError("StochasticArmijoSGD requires a closure that returns loss and computes grads.")

        # compute initial loss + grads
        loss = closure(backward=True)
        loss_value = loss.item()

        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().view(-1))
        if len(grads) == 0:
            raise RuntimeError("No gradients found")
        g_norm_sq = sum(p.grad.detach().pow(2).sum().item() for group in self.param_groups for p in group["params"] if p.grad is not None)

        # store original params
        orig_params = []
        for group in self.param_groups:
            orig_params.append([p.detach().clone() for p in group['params']])

        # line search
        for group, orig_group_params in zip(self.param_groups, orig_params):
            initial_step = group['initial_step']
            c = group['c']
            tau = group['tau']
            max_backtracks = group['max_backtracks']

            step_size = initial_step
            success = False

            for _ in range(max_backtracks):
                # trial update (safe to do without grad tracking)
                with torch.no_grad():
                    for p, orig_p in zip(group['params'], orig_group_params):
                        if p.grad is None:
                            continue
                        p.data.copy_(orig_p - step_size * p.grad)

                # re-evaluate loss (no backward)
                new_loss = closure(backward=False)

                if new_loss.item() <= loss_value - c * step_size * g_norm_sq:
                    success = True
                    break

                step_size *= tau

            if not success:
                # perform update with smallest step size
                continue
                with torch.no_grad():
                    for p , orig_p in zip(group["params"],orig_group_params):
                        if p.grad is None:
                            continue
                        p.data.copy_(orig_p - step_size * p.grad)
        return loss
    
class StochasticArmijoAdam(Optimizer):
    def __init__(self, params, initial_step=1.0, c=0.001, tau=0.5, max_backtracks=20,
                 betas=(0.9, 0.999), eps=1e-8):
        """
        Stochastic Armijo line search optimizer. Adam with Armijo line search.

        Args:
            params: model parameters
            initial_step: initial step size (alpha_0)
            c: Armijo condition parameter (0 < c < 1)
            tau: step size reduction factor (0 < tau < 1)
            max_backtracks: maximum number of backtracking steps
            betas: Adam hyperparameters (β1, β2)
            eps: Adam hyperparameter (for numerical stability)
        """
        defaults = dict(initial_step=initial_step, c=c, tau=tau,
                        max_backtracks=max_backtracks, betas=betas, eps=eps)
        super(StochasticArmijoAdam, self).__init__(params, defaults)

    def step(self, closure):
        if closure is None:
            raise ValueError("StochasticArmijoAdam requires a closure that returns loss and computes grads.")

        # compute initial loss + grads
        loss = closure(backward=True)
        loss_value = loss.item()

        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().view(-1))
        if len(grads) == 0:
            raise RuntimeError("No gradients found")
        g = torch.cat(grads)

        # store original params
        orig_params = []
        for group in self.param_groups:
            orig_params.append([p.detach().clone() for p in group['params']])

        # compute Adam directions
        directions = []
        flat_directions = []
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    directions.append(None)
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                direction = (exp_avg / bias_correction1) / denom

                directions.append(direction.detach().clone())
                flat_directions.append(direction.detach().clone().view(-1))

        d = torch.cat(flat_directions)
        dot_gd = g.dot(d).item()

        # if Adam direction is not descent, fallback to -grad
        if dot_gd >= 0:
            d = -g
            dot_gd = g.dot(d).item()
            # reshape per parameter
            idx = 0
            new_directions = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        new_directions.append(None)
                        continue
                    numel = p.numel()
                    new_directions.append(d[idx:idx + numel].view_as(p))
                    idx += numel
            directions = new_directions

        # line search
        for group, orig_group_params in zip(self.param_groups, orig_params):
            initial_step = group['initial_step']
            c = group['c']
            tau = group['tau']
            max_backtracks = group['max_backtracks']

            step_size = initial_step
            success = False

            for _ in range(max_backtracks):
                with torch.no_grad():
                    for p, orig_p, d_p in zip(group['params'], orig_group_params, directions):
                        if p.grad is None:
                            continue
                        p.data.copy_(orig_p - step_size * d_p)

                new_loss = closure(backward=False)
                print(f"Step size: {step_size}, Loss: {new_loss.item()}")

                if new_loss.item() <= loss_value + c * step_size * dot_gd:
                    success = True
                    break

                step_size *= tau

            if not success:
                with torch.no_grad():
                    for p, orig_p, d_p in zip(group["params"], orig_group_params, directions):
                        if p.grad is None:
                            continue
                        p.data.copy_(orig_p - step_size * d_p)

        return loss

class Armijo_two_way_search(Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        raise NotImplementedError("Two-way Armijo line search not implemented yet.")