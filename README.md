# Efficient Phoneme Decoding for Context-Aware Brain-to-Speech Using Spiking Neural Networks

Current phoneme-level decoders achieve strong performance, yet most require patients to remain physically connected to large computers in clinical settings. Future brain-to-speech systems must be fully implantable, which imposes strict constraints on energy consumption, latency, and causal operation.

Spiking neural networks are well suited to these constraints due to their inherent energy efficiency. However, most existing SNN-based decoding approaches struggle to capture the temporal and contextual dependencies that are intrinsic to speech. Specifically, phoneme production has strong carryover effects in the neural domain, where prior neural states influence subsequent phonemes. 

Motivated by this, we propose a spiking neural network-based brain-to-speech decoder built on an adapted neuron model.
