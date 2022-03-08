#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:discret_policy.py
@time:2021/11/13
"""

import tree
import abc
import numpy as np
import tensorflow as tf
import json
from collections import OrderedDict

try:
    import yaml
except ImportError:
    yaml = None

class BasePolicy:
    def __init__(self,
                 input_demo,
                 output_shape,
                 observation_keys=None,
                 preprocessors=None,
                 name='policy'):
        self._output_shape = output_shape
        self._observation_keys = observation_keys
        self.input_demo = input_demo


        self._name = name


    @property
    def name(self):
        return self._name


    @property
    def observation_keys(self):
        return self._observation_keys

    def reset(self):
        """Reset and clean the policy."""

    def get_weights(self):
        return []

    def set_weights(self, *args, **kwargs):
        return []

    def save_weights(self, *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def weights(self):
        """Returns the list of all policy variables/weights.

        Returns:
          A list of variables.
        """
        return self.trainable_weights + self.non_trainable_weights

    @property
    def trainable_weights(self):
        return []

    @property
    def non_trainable_weights(self):
        return []

    @property
    def variables(self):
        """Returns the list of all policy variables/weights.

        Alias of `self.weights`.

        Returns:
          A list of variables.
        """
        return self.weights

    @property
    def trainable_variables(self):
        return self.trainable_weights

    @property
    def non_trainable_variables(self):
        return self.non_trainable_weights

    @abc.abstractmethod
    def actions(self, inputs):
        """Compute actions for given inputs (e.g. observations)."""
        raise NotImplementedError

    def action(self, *args, **kwargs):
        """Compute an action for a single input, (e.g. observation)."""
        args_, kwargs_ = tree.map_structure(
            lambda x: x[None, ...], (args, kwargs))
        actions = self.actions(*args_, **kwargs_)
        action = tree.map_structure(lambda x: x[0], actions)
        return action

    @abc.abstractmethod
    def log_probs(self, inputs, actions):
        """Compute log probabilities for given actions."""
        raise NotImplementedError

    def log_prob(self, *args, **kwargs):
        """Compute the log probability for a single action."""
        args_, kwargs_ = tree.map_structure(
            lambda x: x[None, ...], (args, kwargs))
        log_probs = self.log_probs(*args_, **kwargs_)
        log_prob = tree.map_structure(lambda x: x[0], log_probs)
        return log_prob

    def actions_and_log_probs(self, *args, **kwargs):
        """Compute actions for given inputs (e.g. observations)."""
        actions = self.actions(*args, **kwargs)
        log_probs = self.log_probs(*args, **kwargs, actions=actions)
        return actions, log_probs

    @abc.abstractmethod
    def probs(self, inputs, actions):
        """Compute probabilities for given actions."""
        raise NotImplementedError

    def prob(self, *args, **kwargs):
        """Compute the probability for a single action."""
        args_, kwargs_ = tree.map_structure(
            lambda x: x[None, ...], (args, kwargs))
        probs = self.probs(*args_, **kwargs_)
        prob = tree.map_structure(lambda x: x[0], probs)
        return prob

    def _filter_observations(self, observations):
        if (isinstance(observations, dict)
                and self._observation_keys is not None):
            observations = type(observations)((
                (key, observations[key])
                for key in self.observation_keys
            ))
        return observations

    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Arguments:
            conditions: Observations to run the diagnostics for.
        Returns:
            diagnostics: OrderedDict of diagnostic information.
        """
        diagnostics = OrderedDict()
        return diagnostics

    def get_diagnostics_np(self, *args, **kwargs):
        diagnostics = self.get_diagnostics(*args, **kwargs)
        diagnostics_np = tree.map_structure(lambda x: x.numpy(), diagnostics)
        return diagnostics_np

    def get_config(self):
        config = {
            'input_shapes': self._input_shapes,
            'output_shape': self._output_shape,
            'observation_keys': self._observation_keys,
            'name': self._name,
        }
        return config

    def _updated_config(self):
        config = self.get_config()
        model_config = {
            'class_name': self.__class__.__name__,
            'config': config,
        }
        return model_config

    def to_yaml(self, **kwargs):
        if yaml is None:
            raise ImportError(
                "Requires yaml module installed (`pip install pyyaml`).")

        yaml.dump(self._updated_config(), **kwargs)

    def to_json(self, **kwargs):
        model_config = self._updated_config()
        return json.dumps(model_config, **kwargs)

    def save(self, filepath, overwrite=True):
        assert overwrite
        config_yaml = self.to_yaml()
        with open(f"{filepath}-config.json", 'w') as f:
            json.dump(config_yaml, f)
        self.save_weights(filepath)

class GaussianPolicy(BasePolicy):
    def __init__(self,hidden_layer_sizes,activation='relu',output_activation='softmax', *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        super(GaussianPolicy, self).__init__(*args, **kwargs)

        self.model = self.feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_shape=(self._output_shape, ),
            activation=self._activation,
            output_activation=self._output_activation
        )
        # 模型初始化
        self.model(self.input_demo)



    def feedforward_model(self,hidden_layer_sizes,
                          output_shape,
                          activation='relu',
                          output_activation='softmax',
                          name='feedforward_model',
                          *args,
                          **kwargs):


        # 构建连续动作空间的神经网络
        output_size = tf.reduce_prod(output_shape)
        print("hidden_layer_sizes:{},output_shape:{},output_size:{}".format(hidden_layer_sizes,output_shape,output_size))
        if 1 < len(output_shape):
            raise NotImplementedError("TODO(hartikainen)")
        model = tf.keras.Sequential((
            *[
                tf.keras.layers.Dense(
                    hidden_layer_size, *args, activation=activation, **kwargs)
                for hidden_layer_size in hidden_layer_sizes
            ],
            tf.keras.layers.Dense(
                output_size, *args, activation=output_activation, **kwargs)
        ), name=name)

        return model


    def actions(self, observations):
        """Compute actions for given observations."""
        action_logits = self.model(observations)
        # 不使用Gumbel-max trick
        actions = tf.argmax(action_logits,axis=-1)

        return actions

    def action(self,observation):
        obs_handle = observation.reshape((1,len(observation)))
        action_logits = self.model(obs_handle)
        # 不使用Gumbel-max trick
        action_id = tf.argmax(action_logits,axis=-1)

        return action_id[0]


    def log_probs(self, observations, actions):
        """Compute log probabilities of `actions` given observations."""
        action_logits = self.model(observations)
        # 不使用Gumbel-max trick
        actions = tf.argmax(action_logits,axis=-1)

        actions_one_hot = tf.one_hot(actions, action_logits.get_shape().as_list()[-1])
        actions_prop = tf.reduce_sum(action_logits * actions_one_hot,axis = 1)  # 要求\pi(a,s)>0
        log_probs = tf.math.log(actions_prop)

        return log_probs

    def probs(self, observations, actions):
        """Compute probabilities of `actions` given observations."""
        action_logits = self.model(observations)
        # 不使用Gumbel-max trick
        actions = tf.argmax(action_logits,axis=-1)

        actions_one_hot = tf.one_hot(actions, action_logits.get_shape().as_list()[-1])
        probs = tf.reduce_sum(action_logits * actions_one_hot,axis = 1)  # 要求\pi(a,s)>0

        return probs

    def actions_and_log_probs(self, observations):
        """Compute actions and log probabilities together.

        We need this functions to avoid numerical issues coming out of the
        squashing bijector (`tfp.bijectors.Tanh`). Ideally this would be
        avoided by using caching of the bijector and then computing actions
        and log probs separately, but that's currently not possible due to the
        issue in the graph mode (i.e. within `tf.function`) bijector caching.
        This method could be removed once the caching works. For more, see:
        https://github.com/tensorflow/probability/issues/840
        """

        action_logits = self.model(observations)
        # print("observations:{},action_logits:{}".format(observations,action_logits))
        # 不使用Gumbel-max trick
        actions = tf.argmax(action_logits,axis=-1)
        actions_one_hot = tf.one_hot(actions, action_logits.get_shape().as_list()[-1])
        actions_prop = tf.reduce_sum(action_logits * actions_one_hot,axis = 1)
        log_probs = tf.math.log(actions_prop)
        # print("actions:{},actions_prop:{}".format(actions,actions_prop))

        return actions, log_probs

    def actions_and_probs(self, observations):
        """Compute actions and probabilities together.

        We need this functions to avoid numerical issues coming out of the
        squashing bijector (`tfp.bijectors.Tanh`). Ideally this would be
        avoided by using caching of the bijector and then computing actions
        and probs separately, but that's currently not possible due to the
        issue in the graph mode (i.e. within `tf.function`) bijector caching.
        This method could be removed once the caching works. For more, see:
        https://github.com/tensorflow/probability/issues/840
        """
        action_logits = self.model(observations)
        # 不使用Gumbel-max trick
        actions = tf.argmax(action_logits,axis=-1)
        actions_one_hot = tf.one_hot(actions, action_logits.get_shape().as_list()[-1])
        probs = tf.reduce_sum(action_logits * actions_one_hot,axis = 1)

        return actions, probs

    def save_weights(self, *args, **kwargs):
        return self.model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.model.load_weights(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.model.set_weights(*args, **kwargs)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model.non_trainable_weights

    @tf.function(experimental_relax_shapes=True)
    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        actions, log_pis = self.actions_and_log_probs(inputs)

        return OrderedDict((
            ('entropy-mean', tf.reduce_mean(-log_pis)),
            ('entropy-std', tf.math.reduce_std(-log_pis)),

            ('actions-mean', tf.reduce_mean(actions)),
            ('actions-std', tf.math.reduce_std(actions)),
            ('actions-min', tf.reduce_min(actions)),
            ('actions-max', tf.reduce_max(actions)),
        ))

if __name__ == '__main__':
    # 处理后的observation
    observation = np.concatenate(([2.0], [1.0]))
    Action_Num = 360
    Input_Demo = observation.reshape((1,len(observation)))
    policy = GaussianPolicy(input_demo=Input_Demo,output_shape=Action_Num,hidden_layer_sizes=(50,50))
    print(policy.actions(Input_Demo))



