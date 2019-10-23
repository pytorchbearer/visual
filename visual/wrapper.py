from visual.redirect_relu import RedirectReLUs
from visual.models.utils import IntermediateLayerGetter


def wrap(model):
    """ Wrap model to prepare for visualisations. Redirects ReLU modules for the first 16 batches and stores references
    to all intermediate layers such that they can be retrieved for ascending.

    Args:
        model: Torch model

    Returns:
        Wrapped model
    """
    return IntermediateLayerGetter(RedirectReLUs(model))
