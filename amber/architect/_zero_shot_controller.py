"""
Zero-shot controller takes in descriptive features from dataset when calling `.get_action()` method,
then generate new architectures for new tasks using a trained controller based on the data-descriptive
features for the new task.

ZZ, May 13, 2020
"""

from ._general_controller import GeneralController
from .common_ops import create_bias, create_weight
from .buffer import MultiManagerBuffer
import tensorflow as tf
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()
    import tensorflow.compat.v1 as tf
import sys


class ZeroShotController(GeneralController):
    def __init__(self, data_description_len, *args, **kwargs):
        self.data_description_len = data_description_len
        super().__init__(*args, **kwargs)
        assert isinstance(self.buffer, MultiManagerBuffer), "ZeroShotController must have MultiManagerBuffer;" \
                                                            " got %s" % self.buffer

    # override
    def _create_weight(self):
        super()._create_weight()
        with tf.variable_scope("description_features", initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)):
            self.data_descriptive_feature = tf.placeholder(shape=(None, self.data_description_len),
                                                           dtype=tf.float32, name='description_features')
            self.w_dd = create_weight(name="w_dd", shape=(self.data_description_len, self.lstm_size))
            self.b_dd = create_bias(name="b_dd", shape=(self.lstm_size,))
            self.g_emb = tf.matmul(self.data_descriptive_feature, self.w_dd) + self.b_dd  # shape: none, lstm_size

    # overwrite
    def get_action(self, description_feature, *args, **kwargs):
        feed_dict = {self.data_descriptive_feature: description_feature}
        probs, onehots = self.session.run([self.sample_probs, self.sample_arc], feed_dict=feed_dict)
        return onehots, probs

    # ovewrite
    def store(self, prob, action, reward, description, *args, **kwargs):
        self.buffer.store(prob=prob, action=action, reward=reward, description=description)

    # overwrite
    def train(self, episode, working_dir):
        try:
            self.buffer.finish_path(self.model_space, episode, working_dir)
        except Exception as e:
            print("cannot finish path in buffer because: %s" % e)
            sys.exit(1)
        aloss = 0
        g_t = 0

        for epoch in range(self.train_pi_iter):
            t = 0
            kl_sum = 0
            ent_sum = 0
            # get data from buffer
            for batch_data in self.buffer.get_data(self.batch_size):
                p_batch, a_batch, ad_batch, nr_batch = \
                    [batch_data[x] for x in ['prob', 'action', 'advantage', 'reward']]
                desc_batch = batch_data['description']
                feed_dict = {self.input_arc[i]: a_batch[:, [i]]
                             for i in range(a_batch.shape[1])}
                feed_dict.update({self.advantage: ad_batch})
                feed_dict.update({self.old_probs[i]: p_batch[i]
                                  for i in range(len(self.old_probs))})
                feed_dict.update({self.reward: nr_batch})
                feed_dict.update({self.data_descriptive_feature: desc_batch})

                _ = self.session.run(self.train_op, feed_dict=feed_dict)
                curr_loss, curr_kl, curr_ent = self.session.run([self.loss, self.kl_div, self.ent], feed_dict=feed_dict)
                aloss += curr_loss
                kl_sum += curr_kl
                ent_sum += curr_ent
                t += 1
                g_t += 1

                if kl_sum / t > self.kl_threshold and epoch > 0:
                    print("     Early stopping at step {} as KL(old || new) = ".format(g_t), kl_sum / t)
                    return aloss / g_t

            if epoch % max(1, (self.train_pi_iter // 5)) == 0:
                print("     Epoch: {} Actor Loss: {} KL(old || new): {} Entropy(new) = {}".format(
                    epoch, aloss / g_t,
                    kl_sum / t,
                    ent_sum / t)
                )

        return aloss / g_t
