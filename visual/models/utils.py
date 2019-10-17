from collections import OrderedDict

import torch
from torch import nn
from torch.jit.annotations import Dict

from visual import LAYER_DICT

_relu_classes = [nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.RReLU, nn.PReLU]


class Storer(nn.Module):
    def __init__(self, save_state, name):
        super().__init__()
        self.save_state = save_state
        self.name = name

    def forward(self, x):
        x = storer(self.save_state, self.name, x)
        return x


def storer(state, name, x):
    try:
        state[LAYER_DICT][name] = x
    except:
        state[LAYER_DICT] = {name: x}
    return x


def basemodel(model):
    class BaseModel(model):
        def __init__(self, layer_names, *args, **kwargs):
            super().__init__(*args, **kwargs)
            relus_to_not_inplace(self)
            self.layer_names = layer_names
            self.__name__ = model.__name__
            self.__qualname__ = model.__qualname__

        def get_layer_names(self):
            return self.layer_names

        def _get_name(self):
            return self.__name__

    return BaseModel


def relus_to_not_inplace(model):
    for m in model.modules():
        if type(m) in _relu_classes:
            m.inplace = False


class IntermediateLayerGetter(nn.Module):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
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

    def __init__(self, model, return_layers):
        super(IntermediateLayerGetter, self).__init__()

        self.model = model
        self.layer_names = []
        self.return_layers = return_layers
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

            if nname in self.return_layers:
                module.__setattr__('forward', new_forward(module.forward, self.return_layers[nname]))
            self.layer_names.append(nname)
            if len(list(layer.named_children())) > 0:
                self.recursive_layer_names(module, nname)

    def forward(self, x, state=None):
        self.model(x)
        if state is not None:
            state[LAYER_DICT] = self.out
