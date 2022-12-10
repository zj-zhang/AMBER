
import numpy as np
import tensorflow as tf
import copy
from amber.backend import Operation
from amber import backend as F
from tqdm import tqdm
from ..supernet.tf1_supernet import EnasAnnModelBuilder
from ...architect.commonOps import batchify


class FeatureModel:
    pass

class FeatModelSparseFfnnModelBuilder(EnasAnnModelBuilder):
    def __init__(self, model_space, inputs_op, output_op, model_compile_dict, session, drop_probs=0.1, l1_reg=0, l2_reg=0, with_skip_connection=True, with_input_blocks=True, with_output_blocks=False, controller=None, feature_model=None, feature_model_trainable=None, name='FeatModEnasDAG'):
        assert feature_model is not None, "You must feed a FeatureModel to FeatModelSparseFfnnModelBuilder; if you are looking to omit FeatureModel, use modeler.supernet.EnasAnnModelBuilder instead"
        assert issubclass(type(feature_model), FeatureModel), "feature_model must subclass modeler.sparse_ffnn.FeatureModel"
        super().__init__(
            model_space=model_space,
            inputs_op=inputs_op,
            output_op=output_op,
            model_compile_dict=model_compile_dict,
            session=session,
            drop_probs=drop_probs,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            with_skip_connection=with_skip_connection,
            with_input_blocks=with_input_blocks,
            with_output_blocks=with_output_blocks,
            controller=controller,
            feature_model=feature_model,
            feature_model_trainable=feature_model_trainable,
            name=name,
        )

def get_features_in_block(spatial_outputs, f_kmean_assign):
    n_clusters = np.max(f_kmean_assign) + 1

    block_ops = []
    for c in range(n_clusters):
        i = np.where(f_kmean_assign == c)[0]
        block_ops.append(tf.gather(spatial_outputs, indices=i, axis=-1, batch_dims=0))
    return block_ops


class CnnFeatureModel(FeatureModel):
    """
    Convert the last Conv layer in a Cnn model as features to a Dnn model
    Keep all Tensors for future Operations
    """

    def __init__(self, base_model, feature_assign, session=None, 
                 feature_map_orientation='channels_last',
                 target_layer=None,
                 trainable=None,
                 name='CnnFeatureModel'):
        self.name = name
        layer_dict = {l.name: l for l in base_model.layers}
        self.trainable = False if trainable is None else trainable
        base_model.trainable = self.trainable
        if self.trainable is True:
            self.trainable_var = [
                v
                for layer in base_model.layers
                for v in layer.variables
                if v.trainable is True
            ]
        else:
            self.trainable_var = []

        if target_layer is None:
            target_layer = sorted([k for k in layer_dict if k.startswith('conv')])[-1]

        self.session = session or F.get_session()
        assert type(self.session) is F.SessionType, f"session should be of amber.backend.SessionType, got {type(self.session)}"
        self.base_model = base_model
        # x_inputs: the actual inputs from data
        self.x_inputs = base_model.inputs
        self.spatial_outputs = layer_dict[target_layer].output
        self.total_feature_num = np.prod(self.spatial_outputs.shape[1:]).value
        # because we will mask by channel, we will re-arrange such that features in
        # the same channel will be close together ZZ 2019.12.4
        if feature_map_orientation == 'channels_last':
            self.orient = [0, 2, 1]
        elif feature_map_orientation == 'channels_first':
            self.orient = [0, 1, 2]
        else:
            raise Exception("cannot understand feature_map_orientation: %s" % feature_map_orientation)
        self.outputs = tf.reshape(tf.transpose(self.spatial_outputs, self.orient), [-1, self.total_feature_num])
        self.feature_assign = feature_assign
        self.load_feature_blocks()
        self.data_gen = None
        # _it are Tensor Iterators
        self.x_it = None
        self.y_it = None
        # _ph are Numpy Array for all data
        self.x_ph = None
        self.y_ph = None
        # tensorflow dataset pseudo_input
        self.pseudo_inputs_pipe = None

    def predict(self, x_, batch_size=None, keep_spatial=True, verbose=True):
        batch_size = batch_size or 32
        preds = []
        gen = batchify(x=x_, batch_size=batch_size, shuffle=False, drop_remainder=False)
        if verbose:
            gen = tqdm(gen, total=x_[0].shape[0]//batch_size)
        for x_b in gen:
            if keep_spatial:
                preds.append(self.session.run(self.spatial_outputs, feed_dict={self.x_inputs[i]: x_b[i] for i in range(len(x_b))}))
            else:
                preds.append(self.session.run(self.outputs, feed_dict={self.x_inputs[i]: x_b[i] for i in range(len(x_b))}))
        preds = np.concatenate(preds)
        return preds

    def load_feature_blocks(self):
        f_assign = copy.copy(self.feature_assign)
        block_ops = get_features_in_block(self.spatial_outputs, f_assign)
        self.input_blocks = [tf.reshape(tf.transpose(x, self.orient), [-1, np.prod(x.shape[1:]).value]) for x in
                             block_ops]
        # pseudo_inputs: after tensor processing of x_input, the output is fed as "pseudo"-input into dense nn
        self.pseudo_inputs = tf.concat(self.input_blocks, axis=1)
        # input_node: instances provided for downstream NAS
        self.input_node_for_nas = [
            Operation('input', shape=(self.input_blocks[i].shape[1].value,), name='Input_%i' % i)
            for i in range(len(block_ops))]
        self.f_assign = f_assign
