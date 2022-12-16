from ... import backend as F
from ...backend import Operation, ComputationNode, get_layer_shortname
from ..base import BaseModelBuilder

class BaseEnasConv1dDAG(BaseModelBuilder):
    def __init__(self,
                 model_space,
                 inputs_op,
                 output_op,
                 model_compile_dict,
                 session,
                 with_skip_connection=True,
                 batch_size=128,
                 keep_prob=0.9,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 reduction_factor=4,
                 controller=None,
                 child_train_op_kwargs=None,
                 stem_config=None,
                 data_format='NWC',
                 train_fixed_arc=False,
                 fixed_arc=None,
                 name='EnasDAG',
                 **kwargs):
        """EnasCnnDAG is a DAG model builder for using the weight sharing framework.

        This class deals with the Convolutional neural network.

        Parameters
        ----------
        model_space: amber.architect.ModelSpace
            model space to search network architectures from
        input_node: amber.architect.Operation, or list
            defines the input shapes and activations
        output_node: amber.architect.Operation, or list
            defines the output shapes and activations
        model_compile_dict: dict
            compile dict for child models
        session: amber.backend.Session
            session for building enas DAG
        train_fixed_arc: bool
            boolean indicator for whether is the final stage; if is True, must provide `fixed_arc` and not connect
            to a controller
        fixed_arc: list-like
            the architecture for final stage training
        name: str
            a name identifier for this instance
        """
        input_node = inputs_op
        output_node = output_op
        assert type(input_node) in (Operation, F.TensorType) or len(
            input_node) == 1, "EnasCnnDAG currently does not accept List type of inputs"
        assert type(output_node) in (Operation, F.TensorType) or len(
            output_node) == 1, "EnasCnnDAG currently does not accept List type of outputs"
        self.input_node = input_node
        self.output_node = output_node
        self.num_layers = len(model_space)
        self.model_space = model_space
        self.model_compile_dict = model_compile_dict
        self.session = session
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.with_skip_connection = with_skip_connection
        self.controller = None
        if controller is not None:
            self.set_controller(controller)
        self.child_train_op_kwargs = child_train_op_kwargs
        self.stem_config = stem_config or {}

        self.name = name
        self.batch_size = batch_size
        self.batch_dim = None
        self.reduction_factor = reduction_factor
        self.keep_prob = keep_prob
        self.data_format = data_format
        self.out_filters = None
        self.branches = []
        self.is_initialized = False

        self.add_conv1_under_pool = kwargs.get("add_conv1_under_pool", True)

        self.train_fixed_arc = train_fixed_arc
        self.fixed_arc = fixed_arc
        if self.train_fixed_arc:
            assert self.fixed_arc is not None, "if train_fixed_arc=True, must provide the architectures in `fixed_arc`"
            assert controller is None, "if train_fixed_arc=True, must not provide controller"
            self.skip_max_depth = None

        self._verify_args()
        self.vars = []
        if controller is None:
            self.controller = None
            print("this EnasDAG instance did not connect a controller; pleaes make sure you are only training a fixed "
                  "architecture.")
        else:
            self.controller = controller
            self._build_sample_arc()
        self._build_fixed_arc()
        #F.init_all_params(sess=self.session)

    def _verify_args(self):
        out_filters = []
        pool_layers = []
        for layer_id in range(len(self.model_space)):
            layer = self.model_space[layer_id]
            this_out_filters = [l.Layer_attributes['filters'] for l in layer]
            assert len(
                set(this_out_filters)) == 1, "EnasConv1dDAG only supports one identical number of filters per layer," \
                                             "but found %i different number of filters in layer %s" % \
                                             (len(set(this_out_filters)), layer)
            if len(out_filters) and this_out_filters[0] != out_filters[-1]:
                pool_layers.append(layer_id - 1)

            out_filters.append(this_out_filters[0])
        self.out_filters = out_filters
        self.pool_layers = pool_layers

        # if train fixed arc, avoid building unused skip connections
        # and verify the input fixed_arc
        if self.train_fixed_arc:
            assert self.fixed_arc is not None
            skip_max_depth = {}
            start_idx = 0
            for layer_id in range(len(self.model_space)):
                skip_max_depth[layer_id] = layer_id
                operation = self.fixed_arc[start_idx]
                total_choices = len(self.model_space[layer_id])
                assert 0 <= operation < total_choices, "Invalid operation selection: layer_id=%i, " \
                                                       "operation=%i, model space len=%i" % (
                                                       layer_id, operation, total_choices)
                if layer_id > 0:
                    skip_binary = self.fixed_arc[(start_idx + 1):(start_idx + 1 + layer_id)]
                    skip = [i for i in range(layer_id) if skip_binary[i] == 1]
                    for d in skip:
                        skip_max_depth[d] = layer_id

                start_idx += 1 + layer_id
            print('-' * 80)
            print(skip_max_depth)
            self.skip_max_depth = skip_max_depth

        if type(self.input_node) is list:
            self.input_node = self.input_node[0]

    def set_controller(self, controller):
        assert self.controller is None, "already has inherent controller, disallowed; start a new " \
                                        "EnasCnnDAG instance if you want to connect another controller"
        self.controller = controller
        self.sample_arc = controller.sample_arc
