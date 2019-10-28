import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbearer
from torchbearer.callbacks.imaging import ImagingCallback


class _Wrapper(nn.Module):
    def __init__(self, image, base_model):
        super(_Wrapper, self).__init__()
        self.base_model = base_model
        self.image = image

    def forward(self, _, state):
        x = self.image(_, state)
        try:
            return self.base_model(x, state)
        except TypeError:
            return self.base_model(x)


class Ascent(ImagingCallback):
    """Callback or stand-alone class to perform gradient ascent on an input image.

    Args:
        image (visual.Image): Input image
        criterion (visual.Criterion): Loss criterion for the gradient ascent
        transform: Transform or transforms to apply to image
        verbose (int): If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training
        optimizer (torch.optim.Optimizer): The optimizer used for image parameter updates. If None: Use Adam
        steps (int): Number of gradient ascent steps to run
    """
    def __init__(self, image, criterion, transform=None, verbose=0, optimizer=None, steps=256):
        super(Ascent, self).__init__(transform=transform)
        self.image = image
        self.criterion = criterion
        self.verbose = verbose
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, image.parameters()), lr=0.05) if optimizer is None else optimizer
        self.steps = steps

    @torchbearer.enable_grad()
    def on_batch(self, state):
        training = state[torchbearer.MODEL].training

        @torchbearer.callbacks.on_sample
        def make_eval(_):
            state[torchbearer.MODEL].eval()

        @torchbearer.callbacks.add_to_loss
        def loss(state):
            return - self.criterion(state)

        model = _Wrapper(self.image, state[torchbearer.MODEL])
        trial = torchbearer.Trial(model, self.optimizer, callbacks=[make_eval, loss])
        trial.for_train_steps(self.steps).to(state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
        trial.run(verbose=self.verbose)

        if training:
            state[torchbearer.MODEL].train()

        return model.image.get_valid_image()

    def run(self, model, verbose=2, device='cpu', dtype=torch.float32):
        """Performs the gradient ascent

        Args:
            model (torch.nn.Module): Base PyTorch model
            verbose (int): If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training
            device (str): Device to perform ascent on, e.g. 'cuda' or 'cpu'
            dtype (torch.dtype): Data type of tensors
        """
        old_verbose = self.verbose
        self.verbose = verbose

        state = torchbearer.State()
        state.update({torchbearer.MODEL: model, torchbearer.DEVICE: device, torchbearer.DATA_TYPE: dtype})
        self.process(state)
        self.verbose = old_verbose


class PyramidAscent(Ascent):
    def __init__(self, image, criterion, transform=None, verbose=0, optimizer=None, steps=32, scales=10, scale_factor=1.4):
        super(PyramidAscent, self).__init__(image, criterion, transform=transform, verbose=verbose, optimizer=optimizer,
                                            steps=steps)
        self.scales = scales
        self.scale_factor = scale_factor

        self.pyramid = nn.ParameterList(PyramidAscent._make_pyramid(self.image.get_valid_image(), self.scales, self.scale_factor)[::-1])

    @staticmethod
    def _make_pyramid(image, scales, scale_factor):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Stop the annoying grid_sample warning
            res = [nn.Parameter(image, requires_grad=False)]
            for _ in range(scales - 1):
                res.append(nn.Parameter(F.interpolate(res[-1].unsqueeze(0), scale_factor=1/scale_factor, mode='bilinear').squeeze(0), requires_grad=False))
        return res

    def on_batch(self, state):
        for i in range(len(self.pyramid)):
            new_scale = self.pyramid[i].data.to(state[torchbearer.DEVICE])
            if i > 0:
                image = self.image.get_valid_image().data
                detail = image - self.pyramid[i - 1].data.to(image.device)  # Get detail
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Stop the annoying grid_sample warning
                    detail = F.interpolate(detail.unsqueeze(0), size=(new_scale.size(1), new_scale.size(2)), mode='bilinear').squeeze(0)  # Upsample
                new_scale = new_scale.data + detail  # Add

            self.image = self.image.load_tensor(new_scale)

            # begin{hack}
            old_param_set = set()
            for group in self.optimizer.param_groups:
                old_param_set.update(set(group['params']))

            new_param_set = set()
            new_param_set.update(self.image.parameters())

            self.optimizer.add_param_group({'params': list(new_param_set.difference(old_param_set))})
            # end{hack}

            super(PyramidAscent, self).on_batch(state)
        return self.image.get_valid_image()
