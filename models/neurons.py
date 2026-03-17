# This file contains different versions of tested neuron models
# We use spikingjelly (Fang et al., 2023) for all SNN applications

from spikingjelly.activation_based import neuron

class LTLIFNode(neuron.ParametricLIFNode):
    '''
    LIF node
    '''
    def __init__(
        self,
        init_tau=2.0,
        surrogate_function=None,
        step_mode="m",
        backend="torch",
        v_threshold=1.0,
        v_reset=0.0,
        decay_input=False,
    ):
        
        super().__init__(
            init_tau=init_tau,
            surrogate_function=surrogate_function,
            step_mode=step_mode,
            backend=backend,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=decay_input,
        )
        self.spike_rate = None