import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR


def all_gather(tensor):
    return AllGatherFunction.apply(tensor)


class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)


def make_contiguous(module):
    """Make the model contigous in order to comply with some distributed strategies.
    https://github.com/lucidrains/DALLE-pytorch/issues/330
    """

    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


# cosine scheduler
def cosine_scheduler(base_value, final_value, total_steps, warmup_steps=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps)

    iters = np.arange(total_steps - warmup_steps)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_steps
    return schedule


# learning rate adjusting for others
def adjust_learning_rate_v1(optimizer, fix_lr, current_step, lr_schedule, wd_schedule):
    current_step = (int)(current_step)
    for i, param_group in enumerate(optimizer.param_groups):
        # check for fix learning rate
        if param_group['type'] == 'backbone':
            param_group["lr"] = lr_schedule['backbone'][current_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[current_step]
        elif param_group['type'] == 'projector':
            param_group["lr"] = lr_schedule['projector'][current_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[current_step]
        else:
            param_group['lr'] = fix_lr


# learning rate adjusting for vicreg and barlowtwins
def adjust_learning_rate_v2(optimizer, base_lr, warmup_steps, total_steps, current_step):
    current_step = (int)(current_step)

    if current_step < warmup_steps:
        lr = base_lr * current_step / warmup_steps
    else:
        current_step -= warmup_steps
        total_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * current_steps / total_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
 
    for pg in optimizer.param_groups:
        pg['lr'] *= lr


# clip gradients
def clip_gradients(model, clip):
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def reset_running_stats(model, set_mode=False):
    for m in model.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                child.track_running_stats = set_mode
                child.running_mean = None
                child.running_var = None


