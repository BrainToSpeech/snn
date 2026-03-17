# This file contains different versions of tested neuron models

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import surrogate
import torch.nn.functional as F
import random

# Basic LIF node
class LTLIFNode(neuron.ParametricLIFNode):
    pass