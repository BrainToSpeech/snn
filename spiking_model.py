# This file contains the Spiking Model and all relevant modules
# We use spikingjelly (Fang et al., 2023) for all SNN applications

from spikingjelly.activation_based import layer, surrogate
from torch import nn

def pick_surrogate(surrogate_name: str):
    if surrogate_name == "atan":
        return surrogate.ATan()
    elif surrogate_name == "sigmoid":
        return surrogate.Sigmoid()
    
class CausalTCNBlock(nn.Module):
    pass
    
class SpikingNeuralNet(nn.Module):
    """
    Proposed hybrid architecture:

    Input (T, B, input_size)
      -> Linear projection to hidden_size
      -> causal TCN stack
      -> compact residual SNN core
      -> linear classifier

    """
    def __init__(self, cfg):
        super().__init__()

        setup = cfg.model.setup
        self._step = 0

        hidden_size = setup.hidden_size

        self.input_proj = nn.Linear(setup.input_size, hidden_size)

        self.temporal = nn.Identity()

        self.snn_core = nn.Identity()

        self.classifier = layer.Linear(hidden_size, setup.output_size, step_mode="m")

        print(
            "[SNN] blocks:",
            ["input_proj", "temporal_tcn", f"snn_core_x{setup.snn_core.n_blocks}", "classifier"],
        )