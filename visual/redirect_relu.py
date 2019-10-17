import torch
import torchbearer
from torch import nn


class TemporaryRelu(nn.Module):
    """ Module to wrap ReLU and RedirectedReLU and call the correct on for the current stage of training.
        For the first epoch only the first 16 batches use redirection and all others use standard relu.

        Args:
            old_module: Standard ReLU module
            redirected_module: Redirected Module
            parent: Parent which tracks the stage in progress and sets parent.redirected
    """
    def __init__(self, old_module, redirected_module, parent):
        super().__init__()
        self.redirected_module = redirected_module
        self.old_module = old_module
        self.parent = [parent]
        self.module = old_module

    def __repr__(self):
        return self.module.__repr__()

    def forward(self, x):
        if self.parent[0].redirected:
            self.module = self.redirected_module
        else:
            self.module = self.old_module
        return self.module(x)


class RedirectReLUs(nn.Module):
    """Module that replaces all ReLU or ReLU6 modules in the model with
    `redirected ReLU <https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py>`__
    versions for the first 16 iterations. Note that this doesn't apply to nn.functional ReLUs.

    Example::

        >>> import torchbearer
        >>> from torchbearer import Trial
        >>> from visual import RedirectReLUs
        >>> model = RedirectReLUs(torch.nn.Sequential(torch.nn.ReLU()))
        >>> @torchbearer.callbacks.on_sample
        ... def input_data(state):
        ...     state[torchbearer.X] = torch.rand(1, 1)
        >>> trial = Trial(model, callbacks=[input_data]).for_steps(1).run()
        >>> print(model)
        RedirectReLUs(
          (model): Sequential(
            (0): RedirectedReLU()
          )
        )
        >>> model = RedirectReLUs(torch.nn.Sequential(torch.nn.ReLU()))
        >>> trial = Trial(model, callbacks=[input_data]).for_steps(17).run()
        >>> print(model)
        RedirectReLUs(
          (model): Sequential(
            (0): ReLU()
          )
        )
    """
    def __init__(self, model):
        super(RedirectReLUs, self).__init__()
        self.relu_types = [torch.nn.ReLU]
        self.old_modules = {}
        self.model = self.replace_relu(model)
        self.batch = None
        self.redirected = True

    def replace_relu(self, model):
        for i, m in enumerate(model.children()):
            if type(m) == torch.nn.ReLU:
                self.old_modules[i] = m
                model._modules[str(i)] = TemporaryRelu(m, RedirectedReLU(), self)
            elif type(m) == torch.nn.ReLU6:
                self.old_modules[i] = m
                model._modules[str(i)] = TemporaryRelu(m, RedirectedReLU6(), self)
        return model

    def forward(self, x, state=None):
        if state is not None:
            self.redirected = not (state[torchbearer.EPOCH] != 0 or state[torchbearer.BATCH] >= 16)
        return self.model(x)


class RedirectedReLU(torch.nn.Module):
    """ Module to wrap the redirected ReLU function.
    See `here <https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py>`__
    """
    def forward(self, x):
        return RedirectedReLUFunction.apply(x)


class RedirectedReLU6(torch.nn.Module):
    """ Module to wrap the redirected ReLU6 function.
    See `here <https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py>`__
    """
    def forward(self, x):
        return RedirectedReLU6Function.apply(x)


class RedirectedReLUFunction(torch.autograd.Function):
    """Reimplementation of the redirected ReLU from
    `tensorflows lucid library <https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py>`__.
    """
    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        relu_grad = grad_input.clone()
        relu_grad[input < 0] = 0

        neg_pushing_lower = torch.lt(input, 0) & torch.gt(grad_input, 0)
        redirected_grad = grad_input
        redirected_grad[neg_pushing_lower] = 0

        batch = grad_input.shape[0]
        reshaped_relu_grad = relu_grad.view(batch, -1)
        relu_grad_mag = torch.norm(reshaped_relu_grad, p=2, dim=1)

        result_grad = relu_grad
        result_grad[relu_grad_mag == 0, :] = redirected_grad[relu_grad_mag == 0, :]

        return result_grad


class RedirectedReLU6Function(torch.autograd.Function):
    """Reimplementation of the redirected ReLU6 from
    `tensorflows lucid library <https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py>`__.
    """
    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        return input.clamp(min=0, max=6)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        relu_grad = grad_input.clone()
        relu_grad[input < 0] = 0
        relu_grad[input > 6] = 0

        neg_pushing_lower = torch.lt(input, 0) & torch.gt(grad_input, 0)
        pos_pushing_higher = torch.gt(input, 6) & torch.lt(grad_input, 0)

        redirected_grad = grad_input
        redirected_grad[neg_pushing_lower] = 0
        redirected_grad[pos_pushing_higher] = 0

        batch = grad_input.shape[0]
        reshaped_relu_grad = relu_grad.view(batch, -1)
        relu_grad_mag = torch.norm(reshaped_relu_grad, p=2, dim=1)

        result_grad = relu_grad
        result_grad[relu_grad_mag == 0, :] = redirected_grad[relu_grad_mag == 0, :]

        return result_grad
