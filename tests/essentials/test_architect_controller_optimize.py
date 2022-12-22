"""Test architect optimizer"""

import copy
import logging
import sys
import tempfile
import unittest

import numpy as np
from parameterized import parameterized_class

from amber import architect
from amber import backend as F
# need to test for seamlessly connecting to manager's architectureDecoder as well. FZZ 2022.5.4
from amber.modeler import architectureDecoder
from amber.utils import static_tf as tf
from amber.utils import testing_utils

logging.disable(sys.maxsize)


class TestModelSpace(testing_utils.TestCase):
    def test_conv1d_model_space(self):
        model_space = architect.ModelSpace()
        num_layers = 2
        out_filters = 8
        layer_ops = [
                architect.Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
                architect.Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
                architect.Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
                architect.Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
        ]
        # init
        for i in range(num_layers):
            model_space.add_layer(i, copy.copy(layer_ops))
        self.assertLen(model_space, num_layers)
        self.assertLen(model_space[0], 4)
        # Add layer
        model_space.add_layer(2, copy.copy(layer_ops))
        self.assertLen(model_space, num_layers + 1)
        # Add op
        model_space.add_state(2, architect.Operation('identity', filters=out_filters))
        self.assertLen(model_space[2], 5)
        # Delete op
        model_space.delete_state(2, 4)
        self.assertLen(model_space[2], 4)
        # Delete layer
        model_space.delete_layer(2)
        self.assertLen(model_space, num_layers)


class TestGeneralController(testing_utils.TestCase):
    def setUp(self):
        super(TestGeneralController, self).setUp()
        self.session = F.Session()
        self.model_space, _ = testing_utils.get_example_conv1d_space(num_layers=3, num_pool=2)
        self.controller = architect.GeneralController(
            model_space=self.model_space,
            buffer_type='ordinal',
            with_skip_connection=True,
            kl_threshold=0.05,
            buffer_size=15,
            batch_size=5,
            session=self.session,
            train_pi_iter=2,
            lstm_size=32,
            lstm_num_layers=1,
            lstm_keep_prob=1.0,
            optim_algo="adam",
            skip_target=0.8,
            skip_weight=None,
        )
        self.tempdir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        super(TestGeneralController, self).tearDown()
        self.tempdir.cleanup()

    def _test_get_architecture(self):
        act, prob = self.controller.get_action()
        self.assertIsInstance(act, np.ndarray)
        self.assertIsInstance(prob, list)
        # initial probas should be close to uniform
        i = 0
        for layer_id in range(len(self.model_space)):
            # operation
            pr = prob[i]
            self.assertAllClose(pr.flatten(), [1./len(pr.flatten())] * len(pr.flatten()), atol=0.05)
            # skip connection
            if layer_id > 0:
                pr = prob[i + 1]
                self.assertAllClose(pr.flatten(), [0.5] * len(pr.flatten()), atol=0.05)
                i += 1
            i += 1

    @unittest.skipIf(F.mod_name=='tensorflow_1', "only implemented in dynamic backend")
    def test_optimizer_dynamic(self):
        a1, p1 = self.controller.get_action()
        a2, p2 = self.controller.get_action()
        a_batch = np.array([a1, a2]).T
        p_batch = [np.concatenate(x) for x in zip(*[p1, p2])]
        self.controller._build_trainer(input_arc=a_batch)
        old_log_probs, old_probs = F.to_numpy(self.controller._build_trainer(input_arc=a_batch))
        losses = []
        max_iter = 100
        for i in range(max_iter):
            loss, _, _ = self.controller._build_train_op(
                input_arc=a_batch, 
                advantage=F.Variable([1,-1], trainable=False),
                old_probs=p_batch
            )
            if i % (max_iter//5) == 0:
                losses.append(F.to_numpy(loss))
        new_log_probs, new_probs = F.to_numpy(self.controller._build_trainer(input_arc=a_batch))
        # loss should decrease over time
        self.assertLess(losses[-1], losses[0])
        # 1st index positive reward should decrease/minimize its loss
        self.assertLess(new_log_probs[0], old_log_probs[0])
        # 2nd index negative reward should increase/increase the loss
        self.assertLess(old_log_probs[1], new_log_probs[1])
    
    @unittest.skipIf(F.mod_name!='tensorflow_1', "only implemented in static/TF1 backend")
    def test_optimize_static(self):
        a1, p1 = self.controller.get_action()
        a2, p2 = self.controller.get_action()
        a_batch = np.array([a1, a2])
        p_batch = [np.concatenate(x) for x in zip(*[p1, p2])]
        feed_dict = {self.controller.input_arc[i]: a_batch[:, [i]]
                     for i in range(a_batch.shape[1])}
        feed_dict.update({self.controller.old_probs[i]: p_batch[i]
                    for i in range(len(self.controller.old_probs))})             
        # add a pseudo reward - the first arc is 1. ; second arc is -1.
        feed_dict.update({self.controller.advantage: np.array([1., -1.]).reshape((2, 1))})
        feed_dict.update({self.controller.reward: np.array([1., 1.]).reshape((2, 1))})
        old_loss = self.session.run(self.controller.onehot_log_prob, feed_dict)
        losses = []
        max_iter = 50
        for i in range(max_iter):
            self.session.run(self.controller.train_op, feed_dict=feed_dict)
            if i % (max_iter//5) == 0:
                losses.append(self.session.run(self.controller.loss, feed_dict))
        new_loss = self.session.run(self.controller.onehot_log_prob, feed_dict)
        # loss should decrease over time
        self.assertLess(losses[-1], losses[0])
        # 1st index positive reward should decrease/minimize its loss
        self.assertLess(new_loss[0], old_loss[0])
        # 2nd index negative reward should increase/increase the loss
        self.assertLess(old_loss[1], new_loss[1])

    def test_train(self):
        # random store some entries
        arcs = []
        probs = []
        rewards = []
        for _ in range(10):
            arc, prob = self.controller.get_action()
            arcs.append(arc)
            probs.append(prob)
            rewards.append(np.random.random(1)[0])
        for arc, prob, reward in zip(*[arcs, probs, rewards]):
            self.controller.store(prob=prob, action=arc,
                                    reward=reward)
        # train
        self.controller.buffer.finish_path(global_ep=0, state_space=self.controller.model_space, working_dir=self.tempdir.name)
        old_loss = self.controller.train()
        for arc, prob, reward in zip(*[arcs, probs, rewards]):
            self.controller.store(prob=prob, action=arc,
                                    reward=reward)
        self.controller.buffer.finish_path(global_ep=1, state_space=self.controller.model_space, working_dir=self.tempdir.name)
        new_loss = self.controller.train()
        self.assertLess(new_loss, old_loss)

@unittest.skipIf(F.mod_name!='tensorflow_1', "only implemented in TF1 backend")
class TestOperationController(testing_utils.TestCase):
    def setUp(self):
        super(TestOperationController, self).setUp()
        self.model_space, _ = testing_utils.get_example_conv1d_space()
        self.controller = architect.OperationController(
            model_space=self.model_space,
            controller_units=8,
            kl_threshold=0.05,
            buffer_size=15,
            batch_size=5,
            train_pi_iter=2
        )
        self.tempdir = tempfile.TemporaryDirectory()

    def test_optimize(self):
        seed = np.array([0]*4).reshape((1, 1, 4))
        for _ in range(8):
            act, proba = self.controller.get_action(seed)
            self.controller.store(
                state=seed,
                prob=proba,
                action=act,
                reward=_
            )
        self.controller.buffer.finish_path(global_ep=0, state_space=self.controller.model_space, working_dir=self.tempdir.name)
        self.controller.train()

    def tearDown(self):
        super(TestOperationController, self).tearDown()
        self.tempdir.cleanup()


@parameterized_class(attrs=('controller_getter', 'decoder_getter'), input_values=[
    (architect.MultiInputController, architectureDecoder.MultiIOArchitecture),
    (architect.MultiIOController, architectureDecoder.MultiIOArchitecture)
])
@unittest.skipIf(F.mod_name!='tensorflow_1', "only implemented in TF1 backend")
class TestMultiIOController(testing_utils.TestCase):
    def setUp(self):
        super(TestMultiIOController, self).setUp()
        num_layers = 5
        self.model_space, _ = testing_utils.get_example_conv1d_space(num_layers=num_layers)
        self.sess = F.Session()
        self.controller = self.controller_getter(
            model_space=self.model_space,
            output_block_unique_connection=True,
            session=self.sess
        )
        self.decoder = self.decoder_getter(
            model_space=self.model_space,
            num_inputs=self.controller.num_input_blocks,
            num_outputs=getattr(self.controller, 'num_output_blocks', 1)
        )
        self.tempdir = tempfile.TemporaryDirectory()

    def test_arc_valid(self):
        arc, prob = self.controller.get_action()
        operations, inputs, skips, outputs = self.decoder.decode(arc)
        self.assertLen(operations, len(self.model_space))
        self.assertLen(inputs, len(self.model_space))
        self.assertLen(skips, len(self.model_space)-1)
        # input masking is incomplete
        #if self.controller.input_block_unique_connection is True:
        #    self.assertEqual(np.sum(inputs), self.controller.num_input_blocks)
        if getattr(self.controller, "output_block_unique_connection", False) is True:
            self.assertEqual(np.sum(outputs), self.controller.num_output_blocks)

    @unittest.skipIf(F.mod_name!='tensorflow_1', "only implemented in TF1 backend")
    def test_optimize(self):
        a1, p1 = self.controller.get_action()
        a2, p2 = self.controller.get_action()
        a_batch = np.array([a1, a2])
        p_batch = [np.concatenate(x) for x in zip(*[p1, p2])]
        feed_dict = {self.controller.input_arc[i]: a_batch[:, [i]]
                     for i in range(a_batch.shape[1])}
        feed_dict.update({self.controller.old_probs[i]: p_batch[i]
                    for i in range(len(self.controller.old_probs))})             
        # add a pseudo reward - the first arc is 1. ; second arc is -1.
        feed_dict.update({self.controller.advantage: np.array([1., -1.]).reshape((2, 1))})
        feed_dict.update({self.controller.reward: np.array([1., 1.]).reshape((2, 1))})
        old_loss = self.sess.run(self.controller.onehot_log_prob, feed_dict)
        losses = []
        max_iter = 100
        for i in range(max_iter):
            self.sess.run(self.controller.train_op, feed_dict=feed_dict)
            if i % (max_iter//5) == 0:
                losses.append(self.sess.run(self.controller.loss, feed_dict))
        new_loss = self.sess.run(self.controller.onehot_log_prob, feed_dict)
        # loss should decrease over time
        self.assertLess(losses[-1], losses[0])
        # 1st index positive reward should decrease/minimize its loss
        self.assertLess(new_loss[0], old_loss[0])
        # 2nd index negative reward should increase/increase the loss
        self.assertLess(old_loss[1], new_loss[1])


    def tearDown(self):
        super(TestMultiIOController, self).tearDown()
        self.tempdir.cleanup()
        self.sess.close()

class TestAmbientController(TestMultiIOController):
    def setUp(self):
        data_description_len = 2
        use_ppo_loss = False
        num_layers = 3
        self.description_feature = np.eye(data_description_len)
        self.model_space, layer_sharing = testing_utils.get_example_conv1d_space(num_layers=num_layers)
        self.sess = F.Session()
        self.controller = architect.controller.ZeroShotController(
            data_description_config={
                "length": data_description_len,
                # "hidden_layer": {"units":config_dict.pop("descriptor_h", 8), "activation": "relu"},
                # "regularizer": {"l1": config_dict.pop("descriptor_l1", 1e-8) }
            },
            model_space=self.model_space,
            session=self.sess,
            share_embedding=layer_sharing,
            with_skip_connection=False,
            skip_weight=None,
            lstm_size=64,
            lstm_num_layers=1,
            kl_threshold=0.1,
            train_pi_iter=100,
            optim_algo='adam',
            temperature=2,
            tanh_constant=2,
            buffer_type="MultiManager",
            buffer_size=1,
            batch_size=1,
            use_ppo_loss=use_ppo_loss,
            rescale_advantage_by_reward=False,
            verbose=False
        )
        self.tempdir = tempfile.TemporaryDirectory()
    
    @unittest.skipIf(F.mod_name!='tensorflow_1', "only implemented in TF1 backend")
    def test_train(self):
        # random store some entries
        for _ in range(10):
            for i in range(len(self.description_feature)):
                arc, prob = self.controller.get_action(self.description_feature[[i]])
                self.controller.store(prob=prob, action=arc, description=self.description_feature[[i]],
                                      reward=np.random.random(1)[0], manager_index=i)
        # train
        self.controller.buffer.finish_path(global_ep=0, state_space=self.controller.model_space, working_dir=self.tempdir.name)
        old_loss = self.controller.train()
        new_loss = self.controller.train()
        self.assertLess(new_loss, old_loss)
    
    @unittest.skipIf(F.mod_name!='tensorflow_1', "only implemented in TF1 backend")
    def test_optimize(self):
        a_batch = []
        p_batch = []
        ads = []  # advantages
        dds = []  # data descriptors
        for i in range(len(self.description_feature)):
            # assign arch as reward 1.
            _, prob = self.controller.get_action(self.description_feature[[i]])
            arc = [1, 2, 0] if i == 0 else [2, 0, 1]
            a_batch.append(arc)
            p_batch.append(prob)
            dds.append(self.description_feature[[i]])
            ads.append(1.)
            # flip the arch and assign reward -1.
            a_batch.append(arc[::-1])
            p_batch.append(prob)
            dds.append(self.description_feature[[i]])
            ads.append(-1.)
        a_batch = np.array(a_batch)
        p_batch = [np.concatenate(x) for x in zip(*p_batch)]
        feed_dict = {self.controller.input_arc[i]: a_batch[:, [i]]
                     for i in range(a_batch.shape[1])}
        feed_dict.update({self.controller.data_descriptive_feature: np.concatenate(dds)})
        feed_dict.update({self.controller.advantage: np.array(ads).reshape((-1, 1))})
        feed_dict.update({self.controller.old_probs[i]: p_batch[i]
                          for i in range(len(self.controller.old_probs))})
        feed_dict.update({self.controller.reward: np.ones(len(ads)).reshape((-1, 1))})
        old_loss = self.sess.run(self.controller.onehot_log_prob, feed_dict)
        losses = []
        max_iter = 50
        for i in range(max_iter):
            self.sess.run(self.controller.train_op, feed_dict=feed_dict)
            if i % (max_iter//5) == 0:
                losses.append(self.sess.run(self.controller.loss, feed_dict))
        new_loss = self.sess.run(self.controller.onehot_log_prob, feed_dict)
        # print(a_batch, ads, losses, new_loss, old_loss)
        # loss should decrease over time
        self.assertLess(losses[-1], losses[0])
        # 1st index positive reward should decrease/minimize its loss
        self.assertLess(new_loss[0], old_loss[0])
        self.assertLess(new_loss[2], old_loss[2])
        # 2nd index negative reward should increase/increase the loss
        self.assertLess(old_loss[1], new_loss[1])
        self.assertLess(old_loss[3], new_loss[3])

    def tearDown(self):
        super(TestAmbientController, self).tearDown()
        self.sess.close()
        self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()

