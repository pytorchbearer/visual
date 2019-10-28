import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchbearer
from torchbearer import cite


IMAGE = torchbearer.state_key('image')
""" State key under which to hold the image being ascended on """


_stanley2007compositional = """
@article{stanley2007compositional,
  title={Compositional pattern producing networks: A novel abstraction of development},
  author={Stanley, Kenneth O},
  journal={Genetic programming and evolvable machines},
  volume={8},
  number={2},
  pages={131--162},
  year={2007},
  publisher={Springer}
}
"""


def _correlate_color(image, correlation, max_norm):
    if image.size(0) == 4:
        alpha = image[-1].unsqueeze(0)
        image = image[:-1]
    else:
        alpha = None
    shape = image.shape
    image = image.view(3, -1).permute(1, 0)
    color_correlation_normalized = correlation / max_norm
    image = image.matmul(color_correlation_normalized.to(image.device).t())
    image = image.permute(1, 0).contiguous().view(shape)
    if alpha is not None:
        image = torch.cat((image, alpha), dim=0)
    return image


def _inverse_correlate_color(image, correlation, max_norm):
    if image.size(0) == 4:
        alpha = image[-1].unsqueeze(0)
        image = image[:-1]
    else:
        alpha = None
    shape = image.shape
    image = image.view(3, -1).permute(1, 0)
    color_correlation_normalized = correlation / max_norm
    image = image.matmul(color_correlation_normalized.to('cpu').t().inverse().to(image.device))
    image = image.permute(1, 0).contiguous().view(shape)
    if alpha is not None:
        image = torch.cat((image, alpha), dim=0)
    return image


def _inverse_sigmoid(x, eps=1e-4):
    x.clamp_(0.01, 0.99)
    return ((x / ((1 - x) + eps)) + eps).log()


def _inverse_clamp(x, color_mean, correlate):
    if correlate:
        if x.dim() > 3:
            x[:3] = x[:3] - color_mean
        else:
            x = x - color_mean
    return x


def image(shape, transform=None, correlate=True, fft=True, sigmoid=True, sd=0.01, decay_power=1, requires_grad=True):
    """ Helper function to generate an image with the given parameters

    Args:
        shape (tuple[int]): Shape of the final image.
        transform: Transforms to apply to the image
        correlate (bool): If True, correlate colour channels of the image when loaded.
        fft (bool): If True, image created in fourier domain
        sigmoid (bool): If True, sigmoid the image
        sd (float): Standard deviation of random initialisation of the image
        decay_power (int / float): Rate of decay on the normalising constant in FFT image
        requires_grad (bool): If True, Image tensor requires gradient.

    Returns:

    """
    if not fft:
        img = torch.randn(shape) if sigmoid else torch.rand(shape)
        img = TensorImage(img, transform=transform, correlate=correlate, requires_grad=requires_grad)
    else:
        img = FFTImage(shape, sd=sd, decay_power=decay_power, transform=transform, correlate=correlate, requires_grad=requires_grad)
    img = img.sigmoid() if sigmoid else img.clamp()
    return img


class Image(nn.Module, torchbearer.callbacks.imaging.ImagingCallback):
    """ Base image class which wraps an image tensor with transforms and allow de/correlating colour channels

    Args:
        transform: Transforms to apply to the image
        correlate (bool): If True, correlate colour channels of the image when loaded.
    """

    def on_batch(self, state):
        return self.get_valid_image()

    def __init__(self, transform=None, correlate=True):
        super(Image, self).__init__()

        self.color_correlation_svd_sqrt = nn.Parameter(
            torch.tensor([[0.26, 0.09, 0.02],
                          [0.27, 0.00, -0.05],
                          [0.27, -0.09, 0.03]], dtype=torch.float32),
            requires_grad=False)
        self.max_norm_svd_sqrt = self.color_correlation_svd_sqrt.norm(dim=0).max()
        self.color_mean = nn.Parameter(torch.tensor([0.48, 0.46, 0.41], dtype=torch.float32), requires_grad=False)

        self.transform = transform if transform is not None else lambda x: x

        self.activation = lambda x: x
        self.correlate = correlate
        self.correction = (lambda x: _correlate_color(x, self.color_correlation_svd_sqrt,
                                                      self.max_norm_svd_sqrt)) if correlate else (lambda x: x)

    def with_handler(self, handler, index=None):
        img = self.get_valid_image()

        if img.dim() == 3:
            img = img.unsqueeze(0)
        rng = range(img.size(0)) if index is None else index
        state = {torchbearer.EPOCH: 0}  # Hack, should do this in a better way
        try:
            for i in rng:
                handler(img[i], i, state)
        except TypeError:
            handler(img[rng], rng, state)
        return self

    @property
    def image(self):
        """
        Class property that returns an un-normalised, parameterised image.

        Returns:
            `torch.Tensor`: Image (channels, height, width) in real space
        """
        raise NotImplementedError

    def get_valid_image(self):
        """
        Return a valid (0, 1) representation of this image, following activation function and colour correction.

        Returns:
            `torch.Tensor`: Image (channels, height, width) in real space
        """
        return self.activation(self.correction(self.image))

    def forward(self, _, state):
        image = self.get_valid_image()
        state[IMAGE] = image
        x = self.transform(image).unsqueeze(0)
        state[torchbearer.INPUT] = x
        return x

    def with_activation(self, function):
        self.activation = function
        return self

    def sigmoid(self):
        return self.with_activation(torch.sigmoid)

    def clamp(self, floor=0., ceil=1.):
        scale = ceil - floor

        def clamp(x):
            return ((x.tanh() + 1.) / 2.) * scale + floor
        if self.correlate:
            def activation(x):
                if x.dim() > 3:
                    x[:3] = x[:3] + self.color_mean
                else:
                    x = x + self.color_mean
                return x
            return self.with_activation(lambda x: clamp(activation(x)))
        else:
            return self.with_activation(clamp)

    def load_file(self, file):
        """Load this Image with the contents of the given file.

        Args:
            file (str): The image file to load
        """
        from PIL import Image
        im = Image.open(file)
        tensor = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.
        return self.load_tensor(tensor)

    def load_tensor(self, tensor):
        """Load this Image with the contents of the given tensor.

        Args:
            tensor: The tensor to load
        """
        if 'sigmoid' in self.activation.__name__:
            tensor = _inverse_sigmoid(tensor)
        elif 'clamp' in self.activation.__name__:
            tensor = _inverse_clamp(tensor, self.color_mean, self.correlate)

        if self.correlate:
            tensor = _inverse_correlate_color(tensor, self.color_correlation_svd_sqrt, self.max_norm_svd_sqrt)

        return self._load_inverse(tensor)

    def _load_inverse(self, tensor):
        raise NotImplementedError


class TensorImage(Image):
    """ Wrapper for Image which takes a torch.Tensor.

    Args:
        tensor (`torch.Tensor`): Image tensor
        transform: Transforms to apply to the image
        correlate (bool): If True, correlate colour channels of the image when loaded.
        requires_grad (bool): If True, tensor requires gradient.
    """
    def __init__(self, tensor, transform=None, correlate=True, requires_grad=True):
        super(TensorImage, self).__init__(transform=transform, correlate=correlate)

        self.tensor = nn.Parameter(tensor, requires_grad=requires_grad)

    @property
    def image(self):
        """ Class property that returns the image tensor

        Returns:
            `torch.Tensor`: Image (channels, height, width) in real space
        """
        return self.tensor

    def _load_inverse(self, tensor):
        self.tensor = nn.Parameter(tensor.to(self.tensor.device), requires_grad=self.tensor.requires_grad)
        return self


def fftfreq2d(w, h):
    import numpy as np
    fy = np.fft.fftfreq(h)[:, None]
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return torch.from_numpy(np.sqrt(fx * fx + fy * fy)).float()


class FFTImage(Image):
    """ Wrapper for Image with creates a random image in the fourer domain with the given parameters

    Args:
        shape (tuple[int]): Shape of the final image.
        sd (float): Standard deviation of random initialisation of the image
        decay_power (int / float): Rate of decay on the normalising constant in FFT image
        transform: Transforms to apply to the image
        correlate (bool): If True, correlate colour channels of the image when loaded.
        requires_grad (bool): If True, Image tensor requires gradient.
    """
    def __init__(self, shape, sd=0.01, decay_power=1, transform=None, correlate=True, requires_grad=True):
        super(FFTImage, self).__init__(transform=transform, correlate=correlate)

        self.decay_power = decay_power

        freqs = fftfreq2d(shape[2], shape[1])
        self.scale = FFTImage._scale(shape, freqs, decay_power)

        param_size = [shape[0]] + list(freqs.shape) + [2]
        param = torch.randn(param_size) * sd
        self.param = nn.Parameter(param, requires_grad=requires_grad)

        self._shape = shape

    @staticmethod
    def _scale(shape, freqs, decay_power):
        scale = torch.ones(1) / torch.max(freqs, torch.tensor([1. / max(shape[2], shape[1])], dtype=torch.float32)).pow(decay_power)
        return nn.Parameter(scale * math.sqrt(shape[2] * shape[1]), requires_grad=False)

    @property
    def image(self):
        """ Class property that returns the image in the real domain

        Returns:
            `torch.Tensor`: Image (channels, height, width) in real space
        """
        ch, h, w = self._shape
        spectrum = self.scale.unsqueeze(0).unsqueeze(3) * self.param
        image = torch.irfft(spectrum, 2)
        image = image[:ch, :h, :w] / 4.0
        return image

    def _load_inverse(self, tensor):
        self._shape = list(tensor.shape)
        self.scale = FFTImage._scale(self._shape, fftfreq2d(self._shape[2], self._shape[1]), self.decay_power)
        self.scale.data = self.scale.data.to(self.param.device)
        if self._shape[2] % 2 == 1:
            tensor = torch.cat((tensor, torch.zeros(self._shape[:2] + [1], device=tensor.device)), dim=2)
        tensor = torch.rfft(tensor.to(self.param.device), 2)
        tensor = tensor * 4 / self.scale.unsqueeze(0).unsqueeze(3)
        self.param = nn.Parameter(tensor.to(self.param.device).data, requires_grad=self.param.requires_grad)
        return self


@cite(_stanley2007compositional)
class CPPNImage(Image):
    """Implements a simple Compositional Pattern Producing Network (CPPN), based on the lucid tutorial
    `xy2rgb <https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/xy2rgb.ipynb>`_.
    This is a convolutional network which is given a coordinate system and outputs an image. The size of the input grid
    can then be changed to produce outputs at arbitrary resolutions.

    Args:
        shape (tuple[int]): Shape (channels, height, width) of the final image.
        hidden_channels (int): The number of channels in hidden layers.
        layers (int): The number of convolutional layers.
        activation: The activation function to use (defaults to CPPNImage.Composite).
        normalise (bool): If True (default), add instance norm to each layer.
        correlate (bool): If True, correlate colour channels of the image when loaded.
        transform: Transforms to apply to the image.
    """

    class Composite(nn.Module):
        """Normalised concatenation of atan(x) and atan^2(x) defined in
        `xy2rgb <https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/xy2rgb.ipynb>`_.
        """
        def forward(self, x):
            x = torch.atan(x)
            return torch.cat((x / 0.67, x.pow(2) / 0.6), 1)

    class UnbiasedComposite(nn.Module):
        """Unbiased normalised concatenation of atan(x) and atan^2(x) defined in
        `xy2rgb <https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/xy2rgb.ipynb>`_.
        """
        def forward(self, x):
            x = torch.atan(x)
            return torch.cat((x / 0.67, (x.pow(2) - 0.45) / 0.396), 1)

    class NormalisedReLU(nn.Module):
        """Normalised ReLU function defined in
        `xy2rgb <https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/xy2rgb.ipynb>`_.
        """
        def forward(self, x):
            x = x.relu()
            return (x - 0.4) / 0.58

    class NormalisedLeakyReLU(nn.LeakyReLU):
        """Normalised leaky ReLU function. See
        `torch.nn.LeakyReLU <https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU>`_ for details.

        Args:
            negative_slope (float): Controls the angle of the negative slope.
        """
        def __init__(self, negative_slope=0.01):
            super(CPPNImage.NormalisedLeakyReLU, self).__init__(negative_slope=negative_slope)
            a = np.random.normal(0.0, 1.0, 10**4)
            a = np.maximum(a, 0.0) + negative_slope * np.minimum(a, 0.0)
            self.mean = a.mean()
            self.std = a.std()

        def forward(self, x):
            x = super(CPPNImage.NormalisedLeakyReLU, self).forward(x)
            return (x - self.mean) / self.std

    @staticmethod
    def _make_grid(height, width):
        r = 3. ** 0.5
        x_coord_range = torch.linspace(-r, r, steps=width)
        y_coord_range = torch.linspace(-r, r, steps=height)
        x, y = torch.meshgrid(y_coord_range, x_coord_range)
        return nn.Parameter(torch.stack((x, y), dim=0).unsqueeze(0), requires_grad=False)

    def __init__(self, shape, hidden_channels=24, layers=8, activation=None, normalise=False, correlate=True, transform=None):
        super(CPPNImage, self).__init__(transform=transform, correlate=correlate)
        activation = CPPNImage.Composite() if activation is None else activation

        (self.channels, self.height, self.width) = shape

        self.loc = CPPNImage._make_grid(self.height, self.width)

        convs = []
        act_ch = hidden_channels * activation(torch.zeros(1, 1, 1, 1)).size(1)
        for i in range(layers):
            in_ch = 2 if i == 0 else act_ch
            c = nn.Conv2d(in_ch, hidden_channels, 1)
            c.weight.data.normal_(0, np.sqrt(1.0 / in_ch))
            c.bias.data.zero_()
            convs.append(c)
            if normalise:
                convs.append(nn.InstanceNorm2d(hidden_channels))
            convs.append(activation)
        c = nn.Conv2d(act_ch, self.channels, 1)
        c.weight.data.zero_()
        c.bias.data.zero_()
        convs.append(c)
        self.convs = nn.Sequential(*convs)

    @property
    def image(self):
        img = self.convs(self.loc).squeeze(0)
        return img

    def resize(self, height, width):
        """Return a new version of this CPPNImage that outputs images at a different resolution. The underlying
        convolutional network will be shared across both objects.

        Args:
            height (int): The height (pixels) of the new image.
            width (int): The width (pixels) of the new image.

        Returns:
            A new CPPNImage with the given size.
        """
        import copy
        res = copy.copy(self)  # Shallow copy, just replace the loc tensor
        res.height = height
        res.width = width

        res.loc = CPPNImage._make_grid(res.height, res.width)
        return res.to(self.loc.device)

    def _load_inverse(self, tensor):
        raise NotImplementedError
