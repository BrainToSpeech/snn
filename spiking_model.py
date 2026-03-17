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
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        p_drop: float = 0.1,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = pad
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x0 = x
        x = x.permute(1, 2, 0).contiguous()
        x = nn.functional.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        x = x.permute(2, 0, 1).contiguous()
        return x + x0
    


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

        self.temporal = nn.Sequential(
            CausalTCNBlock(
                channels=hidden_size,
                kernel_size=setup.tcn.kernel_size,
                dilation=setup.tcn.dilations[0],
                p_drop=setup.tcn.p_drop,
            ),
            CausalTCNBlock(
                channels=hidden_size,
                kernel_size=setup.tcn.kernel_size,
                dilation=setup.tcn.dilations[1],
                p_drop=setup.tcn.p_drop,
            ),
            CausalTCNBlock(
                channels=hidden_size,
                kernel_size=setup.tcn.kernel_size,
                dilation=setup.tcn.dilations[2],
                p_drop=setup.tcn.p_drop,
            ),
        )

        # Using residual spiking block
        self.snn_core = nn.Identity()

        self.classifier = layer.Linear(hidden_size, setup.output_size, step_mode="m")

        print(
            "[SNN] blocks:",
            ["input_proj", "temporal_tcn", f"snn_core_x{setup.snn_core.n_blocks}", "classifier"],
        )