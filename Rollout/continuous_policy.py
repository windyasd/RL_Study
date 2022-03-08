#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:continuous_policy.py
@time:2021/11/12
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import json
from collections import OrderedDict
from contextlib import contextmanager

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors
import abc
try:
    import yaml
except ImportError:
    yaml = None
def create_inputs(shapes, dtypes=None):
    """Creates `tf.keras.layers.Input`s based on input shapes.

    Args:
        input_shapes: (possibly nested) list/array/dict structure of
        inputs shapes.

    Returns:
        inputs: nested structure, of same shape as input_shapes, containing
        `tf.keras.layers.Input`s.

    TODO(hartikainen): Need to figure out a better way for handling the dtypes.
    """
    if dtypes is None:
        dtypes = tree.map_structure(lambda _: None, shapes)
    inputs = tree.map_structure_with_path(create_input, shapes, dtypes)

    return inputs

def create_input(path, shape, dtype=None):
    name = "/".join(str(x) for x in path)

    if dtype is None:
        # TODO(hartikainen): This is not a very robust way to handle the
        # dtypes. Need to figure out something better.
        # Try to infer dtype manually
        dtype = (tf.uint8  # Image observation
                 if len(shape) == 3 and shape[-1] in (1, 3)
                 else tf.float32)  # Non-image

    input_ = tf.keras.layers.Input(
        shape=shape,
        name=name,
        dtype=dtype
    )

    return input_

def deserialize(name, custom_objects=None):
    """Returns a preprocessor function or class denoted by input string.

    Arguments:
        name : String

    Returns:
        Preprocessor function or class denoted by input string.

    For example:
    >>> softlearning.preprocessors.get('convnet_preprocessor')
      <function convnet_preprocessor at 0x7fd170125950>
    >>> softlearning.preprocessors.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown preprocessor: abcd

    Args:
      name: The name of the preprocessor.

    Raises:
        ValueError: `Unknown preprocessor` if the input string does not
        denote any defined preprocessor.
    """
    return deserialize_softlearning_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='preprocessor')

def serialize(preprocessor):
    return serialize_softlearning_object(preprocessor)

def apply_preprocessors(preprocessors, inputs):
    tree.assert_same_structure(inputs, preprocessors)
    preprocessed_inputs = tree.map_structure(
        lambda preprocessor, input_: (
            preprocessor(input_) if preprocessor is not None else input_),
        preprocessors,
        inputs,
    )

    return preprocessed_inputs

class BasePolicy:
    def __init__(self,
                 input_shapes,
                 output_shape,
                 observation_keys=None,
                 preprocessors=None,
                 name='policy'):
        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._observation_keys = observation_keys
        self._create_inputs(input_shapes)

        if preprocessors is None:
            preprocessors = tree.map_structure(lambda x: None, input_shapes)

        preprocessors = tree.map_structure_up_to(
            input_shapes, deserialize, preprocessors)

        self._preprocessors = preprocessors

        self._name = name

    def _create_inputs(self, input_shapes):
        self._inputs = create_inputs(input_shapes)

    @property
    def name(self):
        return self._name

    @property
    def preprocessors(self):
        return self._preprocessors

    @property
    def inputs(self):
        return self._inputs

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

    @contextmanager
    def evaluation_mode(self):
        """Sets the policy to function in evaluation mode.

        The behavior depends on the policy class. For example, GaussianPolicy
        can use the mean of its action distribution as actions.

        TODO(hartikainen): I don't like this way of handling evaluation mode
        for the policies. We should instead have two separete policies for
        training and evaluation, and for example instantiate them like follows:

        ```python
        from softlearning import policies
        training_policy = policies.GaussianPolicy(...)
        evaluation_policy = policies.utils.create_evaluation_policy(training_policy)
        ```
        """
        yield

    def _preprocess_inputs(self, inputs):
        if self.preprocessors is None:
            preprocessors = tree.map_structure(lambda x: None, inputs)
        else:
            preprocessors = self.preprocessors

        preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

        preprocessed_inputs = tf.keras.layers.Lambda(
            cast_and_concat
        )(preprocessed_inputs)

        return preprocessed_inputs

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
            'preprocessors': tree.map_structure(
                serialize, self._preprocessors),
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


class ContinuousPolicy(BasePolicy):
    def __init__(self,
                 action_range,
                 *args,
                 squash=True,
                 **kwargs):
        assert (np.all(action_range == np.array([[-1], [1]]))), (
            "The action space should be scaled to (-1, 1)."
            " TODO(hartikainen): We should support non-scaled actions spaces.")
        self._action_range = action_range
        self._squash = squash
        self._action_post_processor = {
            True: tfp.bijectors.Tanh(),
            False: tfp.bijectors.Identity(),
        }[squash]

        return super(ContinuousPolicy, self).__init__(*args, **kwargs)

    def get_config(self):
        base_config = super(ContinuousPolicy, self).get_config()
        config = {
            **base_config,
            'action_range': self._action_range,
            'squash': self._squash,
        }
        return config


class LatentSpacePolicy(ContinuousPolicy):
    def __init__(self, *args, smoothing_coefficient=None, **kwargs):
        super(LatentSpacePolicy, self).__init__(*args, **kwargs)

        assert smoothing_coefficient is None or 0 <= smoothing_coefficient <= 1

        if smoothing_coefficient is not None and 0 < smoothing_coefficient:
            raise NotImplementedError(
                "TODO(hartikainen): Latent smoothing temporarily dropped on tf2"
                " migration. Should add it back. See:"
                " https://github.com/rail-berkeley/softlearning/blob/46374df0294b9b5f6dbe65b9471ec491a82b6944/softlearning/policies/base_policy.py#L80")

        self._smoothing_coefficient = smoothing_coefficient
        self._smoothing_alpha = smoothing_coefficient or 0
        self._smoothing_beta = (
                np.sqrt(1.0 - np.power(self._smoothing_alpha, 2.0))
                / (1.0 - self._smoothing_alpha))
        self._reset_smoothing_x()
        self._smooth_latents = False

    def _reset_smoothing_x(self):
        self._smoothing_x = np.zeros((1, *self._output_shape))

    def reset(self):
        self._reset_smoothing_x()

    def get_config(self):
        base_config = super(LatentSpacePolicy, self).get_config()
        config = {
            **base_config,
            'smoothing_coefficient': self._smoothing_coefficient,
        }
        return config

class GaussianPolicy(LatentSpacePolicy):
    def __init__(self, *args, **kwargs):
        self._deterministic = False

        super(GaussianPolicy, self).__init__(*args, **kwargs)

        self.shift_and_scale_model = self._shift_and_scale_diag_net(
            inputs=self.inputs,
            output_size=np.prod(self._output_shape) * 2)

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self._output_shape),
            scale_diag=tf.ones(self._output_shape))

        raw_action_distribution = tfp.bijectors.Chain((
            ConditionalShift(name='shift'),
            ConditionalScale(name='scale'),
        ))(base_distribution)

        self.base_distribution = base_distribution
        self.raw_action_distribution = raw_action_distribution
        self.action_distribution = self._action_post_processor(
            raw_action_distribution)

    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        """Compute actions for given observations."""
        observations = self._filter_observations(observations)

        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        shifts, scales = self.shift_and_scale_model(observations)

        if self._deterministic:
            actions = self._action_post_processor(shifts)
        else:
            actions = self.action_distribution.sample(
                batch_shape,
                bijector_kwargs={'scale': {'scale': scales},
                                 'shift': {'shift': shifts}})

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        """Compute log probabilities of `actions` given observations."""
        observations = self._filter_observations(observations)

        shifts, scales = self.shift_and_scale_model(observations)

        if self._deterministic:
            log_probs = tf.fill(
                tf.concat((tf.shape(shifts)[:-1], [1]), axis=0), np.inf)
        else:
            log_probs = self.action_distribution.log_prob(
                actions,
                bijector_kwargs={'scale': {'scale': scales},
                                 'shift': {'shift': shifts}}
            )[..., tf.newaxis]

        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def probs(self, observations, actions):
        """Compute probabilities of `actions` given observations."""
        observations = self._filter_observations(observations)
        shifts, scales = self.shift_and_scale_model(observations)

        if self._deterministic:
            probs = tf.fill(
                tf.concat((tf.shape(shifts)[:-1], [1]), axis=0), np.inf)
        else:
            probs = self.action_distribution.prob(
                actions,
                bijector_kwargs={'scale': {'scale': scales},
                                 'shift': {'shift': shifts}}
            )[..., tf.newaxis]

        return probs

    @tf.function(experimental_relax_shapes=True)
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
        observations = self._filter_observations(observations)

        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        shifts, scales = self.shift_and_scale_model(observations)

        if self._deterministic:
            actions = self._action_post_processor(shifts)
            log_probs = tf.fill(
                tf.concat((tf.shape(shifts)[:-1], [1]), axis=0), np.inf)
        else:
            actions = self.action_distribution.sample(
                batch_shape,
                bijector_kwargs={'scale': {'scale': scales},
                                 'shift': {'shift': shifts}})
            log_probs = self.action_distribution.log_prob(
                actions,
                bijector_kwargs={'scale': {'scale': scales},
                                 'shift': {'shift': shifts}}
            )[..., tf.newaxis]

        return actions, log_probs

    @tf.function(experimental_relax_shapes=True)
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
        observations = self._filter_observations(observations)

        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        shifts, scales = self.shift_and_scale_model(observations)
        if self._deterministic:
            actions = self._action_post_processor(shifts)
            probs = tf.fill(
                tf.concat((tf.shape(shifts)[:-1], [1]), axis=0), np.inf)
        else:
            actions = self.action_distribution.sample(
                batch_shape,
                bijector_kwargs={'scale': {'scale': scales},
                                 'shift': {'shift': shifts}})
            probs = self.action_distribution.prob(
                actions,
                bijector_kwargs={'scale': {'scale': scales},
                                 'shift': {'shift': shifts}}
            )[..., tf.newaxis]

        return actions, probs

    @contextmanager
    def evaluation_mode(self):
        """Activates the evaluation mode, resulting in deterministic actions.

        Once `self._deterministic is True` GaussianPolicy will return
        deterministic actions corresponding to the mean of the action
        distribution. The action log probabilities and probabilities will
        always evaluate to `np.inf` in this mode.

        TODO(hartikainen): I don't like this way of handling evaluation mode
        for the policies. We should instead have two separete policies for
        training and evaluation, and for example instantiate them like follows:

        ```python
        from softlearning import policies
        training_policy = policies.GaussianPolicy(...)
        evaluation_policy = policies.utils.create_evaluation_policy(training_policy)
        ```
        """
        self._deterministic = True
        yield
        self._deterministic = False

    def _shift_and_scale_diag_net(self, inputs, output_size):
        raise NotImplementedError

    def save_weights(self, *args, **kwargs):
        return self.shift_and_scale_model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.shift_and_scale_model.load_weights(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.shift_and_scale_model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.shift_and_scale_model.set_weights(*args, **kwargs)

    @property
    def trainable_weights(self):
        return self.shift_and_scale_model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.shift_and_scale_model.non_trainable_weights

    @tf.function(experimental_relax_shapes=True)
    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        shifts, scales = self.shift_and_scale_model(inputs)
        actions, log_pis = self.actions_and_log_probs(inputs)

        return OrderedDict((
            ('shifts-mean', tf.reduce_mean(shifts)),
            ('shifts-std', tf.math.reduce_std(shifts)),
            ('shifts-max', tf.reduce_max(shifts)),
            ('shifts-min', tf.reduce_min(shifts)),

            ('scales-mean', tf.reduce_mean(scales)),
            ('scales-std', tf.math.reduce_std(scales)),
            ('scales-max', tf.reduce_max(scales)),
            ('scales-min', tf.reduce_min(scales)),

            ('entropy-mean', tf.reduce_mean(-log_pis)),
            ('entropy-std', tf.math.reduce_std(-log_pis)),

            ('actions-mean', tf.reduce_mean(actions)),
            ('actions-std', tf.math.reduce_std(actions)),
            ('actions-min', tf.reduce_min(actions)),
            ('actions-max', tf.reduce_max(actions)),
        ))

def cast_and_concat(x):
    x = tree.map_structure(
        lambda element: tf.cast(element, tf.float32), x)
    x = tree.flatten(x)
    x = tf.concat(x, axis=-1)
    return x

class FeedforwardGaussianPolicy(GaussianPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args,
                 **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        super(FeedforwardGaussianPolicy, self).__init__(*args, **kwargs)

    def _shift_and_scale_diag_net(self, inputs, output_size):
        preprocessed_inputs = self._preprocess_inputs(inputs)
        shift_and_scale_diag = feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_shape=(output_size, ),
            activation=self._activation,
            output_activation=self._output_activation
        )(preprocessed_inputs)

        # 将张量拆分
        shift, scale = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=-1)
        )(shift_and_scale_diag)

        # Computes elementwise softplus: softplus(x) = log(exp(x) + 1).
        scale = tf.keras.layers.Lambda(
            lambda x: tf.math.softplus(x) + 1e-5)(scale)
        shift_and_scale_diag_model = tf.keras.Model(inputs, (shift, scale))

        return shift_and_scale_diag_model

    def get_config(self):
        base_config = super(FeedforwardGaussianPolicy, self).get_config()
        config = {
            **base_config,
            'hidden_layer_sizes': self._hidden_layer_sizes,
            'activation': self._activation,
            'output_activation': self._output_activation,
        }
        return config

def feedforward_model(hidden_layer_sizes,
                      output_shape,
                      activation='relu',
                      output_activation='linear',
                      preprocessors=None,
                      name='feedforward_model',
                      *args,
                      **kwargs):
    # 构建连续动作空间的神经网络
    output_size = tf.reduce_prod(output_shape)
    if 1 < len(output_shape):
        raise NotImplementedError("TODO(hartikainen)")
    model = tf.keras.Sequential((
        tfkl.Lambda(cast_and_concat),
        *[
            tf.keras.layers.Dense(
                hidden_layer_size, *args, activation=activation, **kwargs)
            for hidden_layer_size in hidden_layer_sizes
        ],
        tf.keras.layers.Dense(
            output_size, *args, activation=output_activation, **kwargs),
        # tf.keras.layers.Reshape(output_shape),
    ), name=name)

    return model


