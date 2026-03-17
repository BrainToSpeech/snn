# This file contains different versions of tested neuron models
# We use spikingjelly (Fang et al., 2023) for all SNN applications

from spikingjelly.activation_based import neuron

class LTLIFNode(neuron.ParametricLIFNode):
    '''
    LIF node
    '''
    pass