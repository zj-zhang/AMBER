# -*- coding: UTF-8 -*-

import warnings
from .. import backend as F
from .dag import get_dag
from .dag import ComputationNode
from .base import ModelBuilder



class DAGModelBuilder(ModelBuilder):
    def __init__(self, inputs_op, output_op,
                 model_space, model_compile_dict,
                 num_layers=None,
                 with_skip_connection=True,
                 with_input_blocks=True,
                 dag_func=None,
                 *args, **kwargs):

        if type(inputs_op) not in (list, tuple):
            self.inputs_op = [inputs_op]
            #warnings.warn("inputs_op should be list-like; if only one input, try using ``[inputs_op]`` as argument",
            #             stacklevel=2)
        else:
            self.inputs_op = inputs_op
        self.output_op = output_op
        self.model_space = model_space
        self.num_layers = num_layers or len(self.model_space)
        self.model_compile_dict = model_compile_dict
        self.with_input_blocks = with_input_blocks
        self.with_skip_connection = with_skip_connection
        self.dag_func_ = dag_func
        self.dag_func = get_dag(dag_func) if dag_func is not None else DAG

    def __str__(self):
        s = 'DAGModelBuilder with builder %s' % self.dag_func_
        return s

    def __call__(self, arc_seq, *args, **kwargs):
        input_nodes = self._get_input_nodes()
        output_node = self._get_output_node()
        dag = self.dag_func(arc_seq=arc_seq,
                            num_layers=self.num_layers,
                            model_space=self.model_space,
                            input_node=input_nodes,
                            output_node=output_node,
                            with_skip_connection=self.with_skip_connection,
                            with_input_blocks=self.with_input_blocks,
                            *args,
                            **kwargs)
        try:
            model = dag._build_dag()
            model.compile(**self.model_compile_dict)
        except ValueError:
            print(arc_seq)
            raise Exception('above')
        return model

    def _get_input_nodes(self):
        """Convert input Operation to a list of ComputationNode"""
        input_nodes = []
        for node_op in self.inputs_op:
            node = ComputationNode(node_op, node_name=node_op.Layer_attributes['name'])
            input_nodes.append(node)
        return input_nodes

    def _get_output_node(self):
        """Convert output Operation to ComputationNode"""
        if type(self.output_op) is list:
            raise Exception("DAG currently does not accept output_op in List")
        output_node = ComputationNode(self.output_op, node_name='output')
        return output_node

