
"""
Classes for breaking down an architecture sequence into a more structured format for later use
"""


# Author : zzjfrank
# Date : 2020.8.2; revised 2020-10-06

import numpy as np
import copy
from .base import BaseArchitectDecoder

def get_decoder(m):
    if callable(m):
        return m
    elif type(m) is str:
        if m == 'MultiIOArchitecture':
            return MultiIOArchitecture
        elif m == 'ResConvNetArchitecture':
            return ResConvNetArchitecture
        else:
            raise Exception('Unknown decoder %s' % m)        
    else:
        raise Exception('Unknown decoder %s' % m)


class MultiIOArchitecture(BaseArchitectDecoder):
    def __init__(self, model_space, num_inputs, num_outputs):
        self.model_space = model_space
        self.num_layers = len(model_space)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def decode(self, arc_seq):
        """arc_seq is ordered, in each layer, as
        [op, inp_0, inp_1, ..., con_0, con_1, ...,] X n_layers
        [ 
            [layer0_to_out0, layer1_to_out0, ..., ] X n_outs
        ]
        """
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

    def encode(self, operations, inputs, skips, outputs):
        operations_ = copy.copy(operations)
        inputs = inputs.tolist()
        arc_seq = [operations_.pop(0)] + inputs.pop(0)
        for op, inp, sk in zip(operations_, inputs, skips):
            arc_seq.append(op)
            arc_seq.extend(inp)
            arc_seq.extend(sk)
        arc_seq.extend(outputs.flatten())
        return arc_seq

    def sample(self, seed=None):
        np.random.seed(seed)
        ops = []
        for _ in range(self.num_layers):
            ops.append(np.random.randint(len(self.model_space[_])))
        res_con = []
        for _ in range(1, self.num_layers):
            res_con.append( np.random.binomial(n=1, p=0.5, size=_).tolist())
        inputs = np.random.multinomial(n=1, pvals=np.repeat(1, self.num_layers)/self.num_layers, size=self.num_inputs).T
        outputs = np.random.multinomial(n=1, pvals=np.repeat(1, self.num_layers)/self.num_layers, size=self.num_outputs)
        np.random.seed(None)
        return self.encode(operations=ops, inputs=inputs, skips=res_con, outputs=outputs)


class ResConvNetArchitecture(BaseArchitectDecoder):
    def __init__(self, model_space):
        """ResConvNetArchitecture is a class for decoding and encoding neural architectures of convolutional neural
        networks with residual connections

        Parameters
        ----------
        model_space : amber.architect.ModelSpace
            The model space which architectures are being sampled from
        """
        self.model_space = model_space
        self._num_layers = len(self.model_space)

    def decode(self, arc_seq):
        """Decode a sequence of architecture tokens into operations and res-connections
        """
        start_idx = 0
        operations = []
        res_con = []
        for layer_id in range(self._num_layers):
            operations.append(arc_seq[start_idx])
            if layer_id > 0:
                res_con.append(arc_seq[(start_idx+1) : (start_idx + layer_id + 1)])
            start_idx += layer_id + 1
        return operations, res_con

    def encode(self, operations, res_con):
        """Encode operations and residual connections to a sequence of architecture tokens

        This is the inverse function for `decode`

        Parameters
        ----------
        operations : list
            A list of integers for categorically-encoded operations
        res_con : list
            A list of list where each entry is a binary-encoded residual connections
        """
        operations_ = list(operations)
        arc_seq = [operations_.pop(0)]
        for op, res in zip(operations_, res_con):
            arc_seq.append(op)
            arc_seq.extend(res)
        return arc_seq

    def sample(self, seed=None):
        np.random.seed(seed)
        ops = []
        for _ in range(self._num_layers):
            ops.append(np.random.randint(len(self.model_space[_])))
        res_con = []
        for _ in range(1, self._num_layers):
            res_con.append( np.random.binomial(n=1, p=0.5, size=_).tolist())
        np.random.seed(None)
        return self.encode(operations=ops, res_con=res_con)

