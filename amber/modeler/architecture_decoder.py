
"""
Classes for breaking down an architecture sequence into a more structured format for later use
FZZ
2020.8.2
"""

import numpy as np


class MultiIOArchitecture:
    def __init__(self, num_layers, num_inputs, num_outputs):
        #self.model_space = model_space
        #self.num_layers = len(model_space)
        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def decode(self, arc_seq):
        start_idx = 0
        operations = []
        inputs = []
        skips = []
        for layer_id in range(self.num_layers):
            operation = arc_seq[start_idx]
            #operation = self.model_space[layer_id][operation]
            start_idx += 1
            inp = arc_seq[start_idx : (start_idx + self.num_inputs)]
            if layer_id > 0:
                skip = arc_seq[(start_idx + self.num_inputs) : (start_idx + self.num_inputs + layer_id)]
                skips.append(skip)
            operations.append(operation)
            inputs.append(inp)
            start_idx +=  self.num_inputs + layer_id
        inputs = np.asarray(inputs)
        outputs = np.asarray(arc_seq[start_idx::]).reshape((-1, self.num_layers))
        return operations, inputs, skips, outputs

