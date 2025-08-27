
import torch
from torch.optim import Optimizer
class StochasticArmijoSGD(Optimizer):
    def __init__(self, params, initial_step=1.0, c=0.001, tau=0.5, max_backtracks=10):
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
        g = torch.cat(grads)
        g_norm_sq = g.dot(g).item()

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
                with torch.no_grad():
                    for p , orig_p in zip(group["params"],orig_group_params):
                        if p.grad is None:
                            continue
                        p.data.copy_(orig_p - step_size * p.grad)
        return loss