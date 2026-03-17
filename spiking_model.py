# This file contains the Spiking Model and all relevant modules
# We use spikingjelly (Fang et al., 2023) for all SNN applications

from spikingjelly.activation_based import surrogate
from torch import nn

def pick_surrogate(surrogate_name: str):
    if surrogate_name == "atan":
        return surrogate.ATan()
    elif surrogate_name == "sigmoid":
        return surrogate.Sigmoid()
    
class SpikingNeuralNet(nn.Module):
    """
    Proposed hybrid architecture:

    Input (T, B, input_size)
      -> Linear projection to hidden_size
      -> causal TCN stack
      -> compact residual SNN core
      -> linear classifier

    """
    pass