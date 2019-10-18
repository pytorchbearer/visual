from collections import OrderedDict

import torch
from torch import nn

from visual import LAYER_DICT


class IntermediateLayerGetter(nn.Module):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.

    .. Note::
        It is best if using other wrappers such as RedirectedReLUs to only wrap with IntermediateLayerGetter after all other wrappers

    Arguments:
        model (nn.Module): model on which we will extract the features
    Examples::
        >>> import torchvision
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = IntermediateLayerGetter(m, {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> state = {}
        >>> out = new_m(torch.rand(1, 3, 224, 224), state)
        >>> print([(k, v.shape) for k, v in state[LAYER_DICT].items()])
        [('feat1', torch.Size([1, 64, 56, 56])), ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model):
        super(IntermediateLayerGetter, self).__init__()

        self.model = model
        self.layer_names = []
        self.out = OrderedDict()

        self.recursive_layer_names(model, '')

    def recursive_layer_names(self, layer, pre_name):
        for name, module in layer.named_children():
            nname = pre_name + '_' + name if pre_name != '' else name

            def new_forward(old_forward, nname):
                def new_forward_1(*args, **kwargs):
                    o = old_forward(*args, **kwargs)
                    self.out[nname] = o
                    return o
                return new_forward_1

            module.__setattr__('forward', new_forward(module.forward, nname))
            self.layer_names.append(nname)
            if len(list(layer.named_children())) > 0:
                self.recursive_layer_names(module, nname)

    def forward(self, x, state=None):
        self.model(x)
        if state is not None:
            state[LAYER_DICT] = self.out
