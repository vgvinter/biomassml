import torch.nn as nn
import torch
import warnings
import copy
from typing import Sequence

def patch_module(
    module: torch.nn.Module,
    layers: Sequence,
    weight_dropout: float = 0.0,
    inplace: bool = True,
) -> torch.nn.Module:
    """Replace given layers with weight_drop module of that layer.
    Args:
        module : torch.nn.Module
            The module in which you would like to replace dropout layers.
        layers : list[str]
            Name of layers to be replaced from ['Conv', 'Linear', 'LSTM', 'GRU'].
        weight_dropout (float): The probability a weight will be dropped.
        inplace : bool, optional
            Whether to modify the module in place or return a copy of the module.
    Returns:
        torch.nn.Module:
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    if not inplace:
        module = copy.deepcopy(module)
    changed = _patch_layers(module, layers, weight_dropout)
    if not changed:
        warnings.warn("No layer was modified by patch_module!", UserWarning)
    return module

def get_weight_drop_module(name: str, weight_dropout, **kwargs):
    return {"Linear": WeightDropLinear}[name](weight_dropout, **kwargs)



class WeightDropLinear(torch.nn.Linear):
    """
    Thanks to PytorchNLP for the initial implementation
    # code from https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
    of class WeightDropLinear. Their `License
    <https://github.com/PetrochukM/PyTorch-NLP/blob/master/LICENSE>`__.
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.
    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, weight_dropout=0.0, **kwargs):
        wanted = ["in_features", "out_features"]
        kwargs = {k: v for k, v in kwargs.items() if k in wanted}
        super().__init__(**kwargs)
        self._weight_dropout = weight_dropout

    def forward(self, input):
        w = torch.nn.functional.dropout(
            self.weight, p=self._weight_dropout, training=True
        )
        return torch.nn.functional.linear(input, w, self.bias)


def _patch_layers(
    module: torch.nn.Module, layers: Sequence, weight_dropout: float
) -> bool:
    """
    Recursively iterate over the children of a module and replace them if
    they are in the layers list. This function operates in-place.
    """
    changed = False
    for name, child in module.named_children():
        new_module = None
        for layer in layers:
            if isinstance(child, getattr(torch.nn, layer)):
                new_module = get_weight_drop_module(
                    layer, weight_dropout, **child.__dict__
                )
                break

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # The dropout layer should be deactivated to use DropConnect.
        if isinstance(child, torch.nn.Dropout):
            child.p = 0

        # Recursively apply to child.
        changed |= _patch_layers(child, layers, weight_dropout)
    return changed

def _initialize_layers(activation, initialization, layer):
    if activation == "ReLU":
        nonlinearity = "relu"
    elif activation == "LeakyReLU":
        nonlinearity = "leaky_relu"
    else:
        if initialization == "kaiming":
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"

    if initialization == "kaiming":
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
    elif initialization == "xavier":
        nn.init.xavier_normal_(
            layer.weight,
            gain=nn.init.calculate_gain(nonlinearity)
            if activation in ["ReLU", "LeakyReLU"]
            else 1,
        )
    elif initialization == "random":
        nn.init.normal_(layer.weight)


def _linear_dropout_bn(
    activation,
    initialization,
    use_batch_norm,
    in_units,
    out_units,
    dropout,
    use_dropconnect=False,
    bayesian=False,
):
    if isinstance(activation, str):
        _activation = getattr(nn, activation)
    else:
        _activation = activation
    layers = []
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(num_features=in_units))
    if not bayesian:
        linear = nn.Linear(in_units, out_units)
        _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout != 0:
        if not use_dropconnect:
            layers.append(nn.Dropout(dropout))
    return layers
