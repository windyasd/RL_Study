#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:q_value_nwtwork.py
@time:2021/11/12
"""
import tensorflow as tf
import tree
import numpy as np
import abc
from collections import OrderedDict

def create_ensemble_value_function(N, value_fn, *args, **kwargs):
    # TODO(hartikainen): The ensemble Q-function should support the same
    # interface as the regular ones. Implement the double min-thing
    # as a Keras layer.
    value_fns = tuple(value_fn(*args, **kwargs) for i in range(N))
    return value_fns

def double_feedforward_Q_function(*args, **kwargs):
    return create_ensemble_value_function(
        2, feedforward_Q_function, *args, **kwargs)

def feedforward_Q_function(input_demo,
                           *args,
                           preprocessors=None,
                           observation_keys=None,
                           name='feedforward_Q',
                           **kwargs):

    Q_model = feedforward_model(
        *args,
        output_shape=[1],
        name=name,
        **kwargs
    )
    Q_model(input_demo)


    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys, name=name)

    return Q_function

def feedforward_model(hidden_layer_sizes,
                      output_shape,
                      activation='relu',
                      output_activation='linear',
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

class BaseValueFunction:
    def __init__(self, model, observation_keys, name='value_function'):
        self._observation_keys = observation_keys
        self.model = model
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def observation_keys(self):
        return self._observation_keys

    def reset(self):
        """Reset and clean the value function."""

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.model.set_weights(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        self.model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        self.model.load_weights(*args, **kwargs)

    @property
    def weights(self):
        """Returns the list of all policy variables/weights.

        Returns:
          A list of variables.
        """
        return self.trainable_weights + self.non_trainable_weights

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model.non_trainable_weights

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
    def values(self, inputs):
        """Compute values for given inputs, (e.g. observations)."""
        raise NotImplementedError

    def value(self, *args, **kwargs):
        """Compute a value for a single input, (e.g. observation)."""
        args_, kwargs_ = tree.map_structure(
            lambda x: x[None, ...], (args, kwargs))
        values = self.values(*args_, **kwargs_)
        value = tree.map_structure(lambda x: x[0], values)
        return value

    def _filter_observations(self, observations):
        if (isinstance(observations, dict)
                and self._observation_keys is not None):
            observations = type(observations)((
                (key, observations[key])
                for key in self.observation_keys
            ))
        return observations

    def get_diagnostics(self, *inputs):
        """Return loggable diagnostic information of the value function."""
        diagnostics = OrderedDict()
        return diagnostics

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop('model')
        state.update({
            'model_config': model.get_config(),
            'model_weights': model.get_weights(),
        })
        return state

    def __setstate__(self, state):
        model_config = state.pop('model_config')
        model_weights = state.pop('model_weights')
        model = tf.keras.Model.from_config(model_config)
        model.set_weights(model_weights)
        state['model'] = model
        self.__dict__ = state

class StateActionValueFunction(BaseValueFunction):
    def values(self, observations, actions, **kwargs):
        """Compute values given observations."""
        # actions = actions.numpy()
        # print("observations:{},actions:{}".format(observations,actions))
        # print("observations_type:{},actions_type:{}".format(type(observations),type(actions)))
        # print("observations.ndim:{},actions.ndim:{}")
        if actions.ndim<2:
            actions=actions.reshape((len(actions),1))
        if observations.ndim<2:
            observations = observations.reshape((1,len(observations)))
        inputs = np.concatenate((observations,actions),axis=-1)
        # print("inputs:{}".format(inputs))
        values = self.model(inputs)
        return values

if __name__ == '__main__':
    # 处理后的observation
    observation = np.concatenate(([2.0], [1.0]))
    # 动作编号
    action = 0
    input = np.concatenate((observation,[action]))
    input_handle = input.reshape((1,len(input)))
    print(input.shape)
    Qs = double_feedforward_Q_function(input_demo=input,hidden_layer_sizes=(50,50),name = 'Double_Q_Value_Function')
    for Q in Qs:
        print("Q.values(observation,action):{}".format(Q.values(observation,action)))


