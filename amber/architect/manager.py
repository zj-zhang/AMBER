# -*- coding: UTF-8 -*-

import gc
import os
import warnings

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model

from .store import get_store_fn


class BaseNetworkManager:
    def __init__(self, *args, **kwargs):
        # abstract
        pass

    def get_rewards(self, trial, model_arc):
        raise NotImplementedError("Abstract method.")


class GeneralManager(BaseNetworkManager):
    """
    Manager creates subnetworks, training them on a dataset, and retrieving
    rewards.
    """

    def __init__(self,
                 train_data,
                 validation_data,
                 model_fn,
                 reward_fn,
                 store_fn,
                 model_compile_dict,
                 working_dir='.',
                 save_full_model=False,
                 epochs=5,
                 child_batchsize=128,
                 verbose=0,
                 **kwargs):
        super(GeneralManager, self).__init__(**kwargs)
        self.train_data = train_data
        self.validation_data = validation_data
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.model_compile_dict = model_compile_dict

        self.save_full_model = save_full_model
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.verbose = verbose

        self.model_fn = model_fn
        self.reward_fn = reward_fn
        self.store_fn = get_store_fn(store_fn)

    def get_rewards(self, trial, model_arc):
        # print('-'*80, model_arc, '-'*80)
        train_graph = tf.Graph()
        train_sess = tf.Session(graph=train_graph)
        with train_graph.as_default(), train_sess.as_default():
            model = self.model_fn(model_arc)  # a compiled keras Model

            # unpack the dataset
            X_val, y_val = self.validation_data[0:2]
            X_train, y_train = self.train_data

            # train the model using Keras methods
            print(" Trial %i: Start training model..." % trial)
            hist = model.fit(X_train, y_train,
                             batch_size=self.batchsize,
                             epochs=self.epochs,
                             verbose=self.verbose,
                             validation_data=(X_val, y_val),
                             callbacks=[ModelCheckpoint(os.path.join(self.working_dir, 'temp_network.h5'),
                                                        monitor='val_loss', verbose=self.verbose,
                                                        save_best_only=True),
                                        EarlyStopping(monitor='val_loss', patience=5, verbose=self.verbose)]
                             )
            # load best performance epoch in this training session
            model.load_weights(os.path.join(self.working_dir, 'temp_network.h5'))

            # evaluate the model by `reward_fn`
            this_reward, loss_and_metrics, reward_metrics = \
                self.reward_fn(model, (X_val, y_val),
                               session=train_sess,
                               graph=train_graph)
            loss = loss_and_metrics.pop(0)
            loss_and_metrics = {str(self.model_compile_dict['metrics'][i]): loss_and_metrics[i] for i in
                                range(len(loss_and_metrics))}
            loss_and_metrics['loss'] = loss
            if reward_metrics:
                loss_and_metrics.update(reward_metrics)

            # do any post processing,
            # e.g. save child net, plot training history, plot scattered prediction.
            if self.store_fn:
                val_pred = model.predict(X_val)
                self.store_fn(
                    trial=trial,
                    model=model,
                    hist=hist,
                    data=self.validation_data,
                    pred=val_pred,
                    loss_and_metrics=loss_and_metrics,
                    working_dir=self.working_dir,
                    save_full_model=self.save_full_model,
                    knowledge_func=self.reward_fn.knowledge_function
                )

        # clean up resources and GPU memory
        del model
        del hist
        gc.collect()
        return this_reward, loss_and_metrics


class EnasManager(GeneralManager):
    def __init__(self, session=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if session is None:
            self.session = self.model_fn.session
        else:
            self.session = session
        self.model = None
        self.disable_controller = kwargs.pop("disable_controller", False)

    def get_rewards(self, trial, model_arc=None, nsteps=None):
        """
        Because Enas will train child model by random sampling an architecture to activate for each mini-batch,
        there will not be any rewards evaluation in the Manager anymore.
        However, we can still use `get_rewards` as a proxy to train child models
        Args:
            trial:
            model_arc:
            nsteps: optional, if specified, train model nsteps of batches instead of a whole epoch
        Returns:

        """
        if self.model is None:
            self.model = self.model_fn()

        # just playing around for evaluating VC dimensions by fitting random noise- remember to remove these lines
        # ZZ 2020.2.4
        # if self.disable_controller:
        #    this_reward = np.random.uniform(0,1)
        #    #if model_arc is not None and model_arc[0] == 0:
        #    #    this_reward = np.random.uniform(0.9,1)
        #    loss_and_metrics = {'loss': this_reward}
        #    return this_reward, loss_and_metrics
        # end
        if model_arc is None:
            # unpack the dataset
            X_val, y_val = self.validation_data[0:2]
            X_train, y_train = self.train_data
            # train the model using EnasModel methods
            print(" Trial %i: Start training model with sample_arc..." % trial)
            hist = self.model.fit(X_train, y_train,
                                  batch_size=self.batchsize,
                                  nsteps=nsteps,
                                  epochs=self.epochs,
                                  verbose=self.verbose,
                                  # comment out because of temporary
                                  # incompatibility with tf.data.Dataset
                                  # validation_data=(X_val, y_val),
                                  )

            # do any post processing,
            # e.g. save child net, plot training history, plot scattered prediction.
            if self.store_fn:
                val_pred = self.model.predict(X_val)
                self.store_fn(
                    trial=trial,
                    model=self.model,
                    hist=hist,
                    data=self.validation_data,
                    pred=val_pred,
                    loss_and_metrics=None,
                    working_dir=self.working_dir,
                    save_full_model=self.save_full_model,
                    knowledge_func=self.reward_fn.knowledge_function
                )
            return None, None
        else:
            model = self.model_fn(model_arc)
            this_reward, loss_and_metrics, reward_metrics = \
                self.reward_fn(model, self.validation_data,
                               session=self.session)
            loss = loss_and_metrics.pop(0)
            loss_and_metrics = {str(self.model_compile_dict['metrics'][i]): loss_and_metrics[i] for i in
                                range(len(loss_and_metrics))}
            loss_and_metrics['loss'] = loss
            if reward_metrics:
                loss_and_metrics.update(reward_metrics)
            # enable this to overwrite a random reward when disable controller
            if self.disable_controller:
                this_reward = np.random.uniform(0, 1)
            # end
            return this_reward, loss_and_metrics


class NetworkManager(BaseNetworkManager):
    """
    DEPRECATION WARNING: will be deprecated in the future; please `GeneralManager`
    Helper class to manage the generation of subnetwork training given a dataset

    Manager creates subnetworks, training them on a dataset, and retrieving
    rewards in the term of accuracy, which is passed to the controller RNN.

    Parameters
    ----------
    input_state: tuple
        specify the input shape to `model_fn`
    output_state: str
        parsed to `get_layer` for a fixed output layer
    model_fn: callable
        a function for creating Keras.Model; takes model_states, input_state, output_state, model_compile_dict
    reward_fn: callable
        a function for computing Reward; takes two arguments, model and data
    store_fn: callable
        a function for processing/plotting trained child model
    epochs: int
        number of epochs to train the subnetworks
    child_batchsize: int
        batchsize of training the subnetworks
    acc_beta: float
        exponential weight for the accuracy
    clip_rewards: float
        to clip rewards in [-range, range] to prevent large weight updates. Use when training is highly unstable.
    """

    def __init__(self,
                 train_data,
                 validation_data,
                 input_state,
                 output_state,
                 model_compile_dict,
                 model_fn,
                 reward_fn,
                 store_fn,
                 working_dir='.',
                 save_full_model=False,
                 train_data_size=None,
                 epochs=5,
                 child_batchsize=128,
                 tune_data_feeder=False,
                 verbose=0):

        super(NetworkManager, self).__init__()
        warnings.warn("`NetworkManager` will be deprecated in the future; please `GeneralManager`")
        self.train_data = train_data
        self.validation_data = validation_data
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        self.save_full_model = save_full_model
        self.train_data_size = len(train_data[0]) if train_data_size is None else train_data_size
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.tune_data_feeder = tune_data_feeder
        self.verbose = verbose
        self.input_state = input_state
        self.output_state = output_state
        self.model_compile_dict = model_compile_dict

        self.model_fn = model_fn
        self.reward_fn = reward_fn
        self.store_fn = store_fn

        self.entropy_record = []
        # if tune data feeder, must NOT provide train data
        assert self.tune_data_feeder == (self.train_data is None)

    def get_rewards(self, trial, model_states):
        """
        Creates a subnetwork given the model_states predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            model_states: a list of parsed model_states obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `model_states` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `model_states` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

        Returns:
            a reward for training a model with the given model_states
        """
        train_graph = tf.Graph()
        train_sess = tf.Session(graph=train_graph)
        with train_graph.as_default(), train_sess.as_default():

            if self.tune_data_feeder:
                feeder_size = model_states[0].Layer_attributes['size']
                if self.verbose: print("feeder_size", feeder_size)

            if self.tune_data_feeder:
                model, feeder = self.model_fn(model_states, self.input_state, self.output_state,
                                              self.model_compile_dict)  # type: Model - compiled
            else:
                model = self.model_fn(model_states, self.input_state, self.output_state,
                                      self.model_compile_dict)  # type: Model - compiled

            # unpack the dataset
            X_val, y_val = self.validation_data[0:2]

            # model fitting
            if self.tune_data_feeder:  # tuning data generator feeder
                hist = model.fit_generator(feeder,
                                           epochs=self.epochs,
                                           steps_per_epoch=self.train_data_size // feeder_size,
                                           verbose=self.verbose,
                                           validation_data=(X_val, y_val),
                                           callbacks=[ModelCheckpoint(os.path.join(self.working_dir, 'temp_network.h5'),
                                                                      monitor='val_loss', verbose=self.verbose,
                                                                      save_best_only=True),
                                                      EarlyStopping(monitor='val_loss', patience=5,
                                                                    verbose=self.verbose)],
                                           use_multiprocessing=True,
                                           max_queue_size=8)
            else:
                if type(self.train_data) == tuple or type(self.train_data) == list:
                    # is a tuple/list of np array
                    X_train, y_train = self.train_data

                    # train the model using Keras methods
                    print("	Start training model...")
                    hist = model.fit(X_train, y_train,
                                     batch_size=self.batchsize,
                                     epochs=self.epochs,
                                     verbose=self.verbose,
                                     validation_data=(X_val, y_val),
                                     callbacks=[ModelCheckpoint(os.path.join(self.working_dir, 'temp_network.h5'),
                                                                monitor='val_loss', verbose=self.verbose,
                                                                save_best_only=True),
                                                EarlyStopping(monitor='val_loss', patience=5, verbose=self.verbose)]
                                     )
                else:
                    # is a generator
                    hist = model.fit_generator(self.train_data,
                                               epochs=self.epochs,
                                               steps_per_epoch=self.train_data_size // self.batchsize,
                                               verbose=self.verbose,
                                               validation_data=(X_val, y_val),
                                               callbacks=[
                                                   ModelCheckpoint(os.path.join(self.working_dir, 'temp_network.h5'),
                                                                   monitor='val_loss',
                                                                   verbose=self.verbose,
                                                                   save_best_only=True),
                                                   EarlyStopping(monitor='val_loss', patience=5, verbose=self.verbose)],
                                               use_multiprocessing=True,
                                               max_queue_size=8)

            # load best performance epoch in this training session
            model.load_weights(os.path.join(self.working_dir, 'temp_network.h5'))

            # evaluate the model by `reward_fn`
            this_reward, loss_and_metrics, reward_metrics = self.reward_fn(model, (X_val, y_val), session=train_sess)
            loss = loss_and_metrics.pop(0)
            loss_and_metrics = {str(self.model_compile_dict['metrics'][i]): loss_and_metrics[i] for i in
                                range(len(loss_and_metrics))}
            loss_and_metrics['loss'] = loss
            if reward_metrics:
                loss_and_metrics.update(reward_metrics)

            # do any post processing,
            # e.g. save child net, plot training history, plot scattered prediction.
            if self.store_fn:
                val_pred = model.predict(X_val)
                self.store_fn(
                    trial=trial,
                    model=model,
                    hist=hist,
                    data=self.validation_data,
                    pred=val_pred,
                    loss_and_metrics=loss_and_metrics,
                    working_dir=self.working_dir,
                    save_full_model=self.save_full_model
                )

        # clean up resources and GPU memory
        del model
        del hist
        gc.collect()
        return this_reward, loss_and_metrics
