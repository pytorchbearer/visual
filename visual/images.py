import math

import numpy as np

import torch
import torch.nn as nn

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
    image = image.matmul(color_correlation_normalized.t())
    image = image.permute(1, 0).contiguous().view(shape)
    if alpha is not None:
        image = torch.cat((image, alpha), dim=0)
    return image


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

        ch, h, w = shape
        freqs = fftfreq2d(w, h)
        scale = torch.ones(1) / torch.max(freqs, torch.tensor([1. / max(w, h)], dtype=torch.float32)).pow(decay_power)
        self.scale = nn.Parameter(scale * math.sqrt(w * h), requires_grad=False)

        param_size = [ch] + list(freqs.shape) + [2]
        param = torch.randn(param_size) * sd
        self.param = nn.Parameter(param, requires_grad=requires_grad)

        self._shape = shape

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

    def __init__(self, shape, hidden_channels=24, layers=8, activation=None, normalise=True, correlate=True, transform=None):
        super(CPPNImage, self).__init__(transform=transform, correlate=correlate)
        activation = CPPNImage.Composite() if activation is None else activation
        r = 3. ** 0.5
        (self.channels, self.height, self.width) = shape

        x_coord_range = torch.linspace(-r, r, steps=self.width)
        y_coord_range = torch.linspace(-r, r, steps=self.height)
        x, y = torch.meshgrid(y_coord_range, x_coord_range)

        self.loc = nn.Parameter(torch.stack((x, y), dim=0).unsqueeze(0), requires_grad=False)

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

    def resize_as(self, height, width):
        """Return a new version of this CPPNImage that outputs images at a different resolution. The underlying
        convolutional network will be shared across both objects.

        Args:
            height (int): The height (pixels) of the new image.
            width (int): The width (pixels) of the new image.

        Returns:
            A new CPPNImage with the given size.
        """
        res = CPPNImage((self.channels, height, width), correlate=self.correlate, transform=self.transform)
        res.convs = self.convs
        return res
