#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:SAC_Algorithm.py
@time:2021/11/13
"""
import numpy as np
import tensorflow as tf
from numbers import Number
import inspect
import tree

from copy import deepcopy
from collections import OrderedDict
from numbers import Number
import abc
import numpy as np
import tensorflow as tf
from itertools import count
import tensorflow_probability as tfp
import gtimer as gt
from tensorflow.python.training.tracking.tracking import AutoTrackable
import math
from rollout import rollouts


def td_targets(rewards, discounts, next_values):
    return rewards + discounts * next_values


def compute_Q_targets(next_Q_values,
                      next_log_pis,
                      rewards,
                      terminals,
                      discount,
                      entropy_scale,
                      reward_scale):
    # 这个函数计算了 U_t^{(q)}
    # 这里根据Q-function 和 policy 计算出了 V-function
    next_values = next_Q_values - entropy_scale * next_log_pis
    terminals = tf.cast(terminals, next_values.dtype)

    Q_targets = td_targets(
        rewards=reward_scale * rewards,
        discounts=discount,
        next_values=(1.0 - terminals) * next_values)

    return Q_targets

class RLAlgorithm(AutoTrackable):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            pool,
            sampler,
            n_epochs=1000,
            train_every_n_steps=1,
            n_train_repeat=1,
            min_pool_size=1,
            batch_size=1,
            max_train_repeat_per_timestep=5,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_render_kwargs=None,
            num_warmup_samples=0,
    ):
        """
        Args:
            pool (`ReplayPool`): Replay pool to add gathered samples to.
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            epoch_length (`int`): Epoch length. 每个epoch采样的个数
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_render_kwargs (`None`, `dict`): Arguments to be passed for
                rendering evaluation rollouts. `None` to disable rendering.
            num_warmup_samples ('int'): Number of random samples to warmup the
                replay pool with.
        """
        self.sampler = sampler
        self.pool = pool

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size
        self._max_train_repeat_per_timestep = max(
            max_train_repeat_per_timestep, n_train_repeat)
        self._train_every_n_steps = train_every_n_steps
        self._epoch_length = epoch_length

        self._eval_n_episodes = eval_n_episodes

        self._eval_render_kwargs = eval_render_kwargs or {}


        self._epoch = 0
        self._timestep = 0
        self._num_train_steps = 0


    def _training_after_hook(self):
        """Method called after the actual training loops."""
        pass

    def _timestep_before_hook(self, *args, **kwargs):
        """Hook called at the beginning of each timestep."""
        pass

    def _timestep_after_hook(self, *args, **kwargs):
        """Hook called at the end of each timestep."""
        pass

    def _epoch_before_hook(self):
        """Hook called at the beginning of each epoch."""
        self._train_steps_this_epoch = 0

    def _epoch_after_hook(self, *args, **kwargs):
        """Hook called at the end of each epoch."""
        pass

    def _training_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        return self.pool.random_batch(batch_size, **kwargs)

    def _evaluation_batch(self, *args, **kwargs):
        return self._training_batch(*args, **kwargs)

    @property
    def _training_started(self):
        return self._total_timestep > 0

    @property
    def _total_timestep(self):
        total_timestep = self._epoch * self._epoch_length + self._timestep
        return total_timestep

    def train(self, *args, **kwargs):
        """Initiate training of the SAC instance."""
        self._train()
        # return self._train()

    def _train(self):
        """Return a generator that runs the standard RL loop."""
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)


        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            update_diagnostics = []

            start_samples = self.sampler._total_samples
            for i in count():
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                if (samples_now >= start_samples + self._epoch_length
                        and self.ready_to_train):
                    break

                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                self._do_sampling(timestep=self._total_timestep)
                gt.stamp('sample')

                # 判断pool中的sample个数是否满足最小训练需求
                if self.ready_to_train:
                    repeat_diagnostics = self._do_training_repeats(
                        timestep=self._total_timestep)
                    if repeat_diagnostics is not None:
                        update_diagnostics.append(repeat_diagnostics)

                gt.stamp('train')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            update_diagnostics = tree.map_structure(
                lambda *d: np.mean(d), *update_diagnostics)

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            gt.stamp('training_paths')
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment)
            gt.stamp('evaluation_paths')

            training_metrics = self._evaluate_rollouts(
                training_paths,
                training_environment,
                self._total_timestep,
                evaluation_type='train')
            gt.stamp('training_metrics')
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths,
                    evaluation_environment,
                    self._total_timestep,
                    evaluation_type='evaluation')
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}

            self._epoch_after_hook(training_paths)
            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = {
                key: times[-1]
                for key, times in gt.get_times().stamps.itrs.items()
            }

            # TODO(hartikainen/tf2): Fix the naming of training/update
            # diagnostics/metric
            diagnostics.update((
                ('evaluation', evaluation_metrics),
                ('training', training_metrics),
                ('update', update_diagnostics),
                ('times', time_diagnostics),
                ('sampler', sampler_diagnostics),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('total_timestep', self._total_timestep),
                ('num_train_steps', self._num_train_steps),
            ))

        self.sampler.terminate()

        self._training_after_hook()

    def _evaluation_paths(self, policy, evaluation_env):
        if self._eval_n_episodes < 1: return ()

        # TODO(hartikainen): I don't like this way of handling evaluation mode
        # for the policies. We should instead have two separete policies for
        # training and evaluation.
        with policy.evaluation_mode():
            paths = rollouts(
                self._eval_n_episodes,
                evaluation_env,
                policy,
                self.sampler._max_path_length,
                render_kwargs=self._eval_render_kwargs)

        return paths

    def _evaluate_rollouts(self,
                           episodes,
                           environment,
                           timestep,
                           evaluation_type=None):
        """Compute evaluation metrics for the given rollouts."""

        episodes_rewards = [episode['rewards'] for episode in episodes]
        episodes_reward = [np.sum(episode_rewards)
                           for episode_rewards in episodes_rewards]
        episodes_length = [episode_rewards.shape[0]
                           for episode_rewards in episodes_rewards]

        diagnostics = OrderedDict((
            ('episode-reward-mean', np.mean(episodes_reward)),
            ('episode-reward-min', np.min(episodes_reward)),
            ('episode-reward-max', np.max(episodes_reward)),
            ('episode-reward-std', np.std(episodes_reward)),
            ('episode-length-mean', np.mean(episodes_length)),
            ('episode-length-min', np.min(episodes_length)),
            ('episode-length-max', np.max(episodes_length)),
            ('episode-length-std', np.std(episodes_length)),
        ))

        environment_infos = environment.get_path_infos(
            episodes, timestep, evaluation_type=evaluation_type)
        diagnostics['environment_infos'] = environment_infos

        return diagnostics

    @abc.abstractmethod
    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        raise NotImplementedError

    @property
    def ready_to_train(self):
        return self._min_pool_size <= self.pool.size

    def _do_sampling(self, timestep):
        self.sampler.sample()

    def _do_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        if timestep % self._train_every_n_steps > 0: return
        # 避免数据很少的时候训练太多步
        trained_enough = (
                self._train_steps_this_epoch
                > self._max_train_repeat_per_timestep * self._timestep)
        if trained_enough: return

        diagnostics = [
            self._do_training(iteration=timestep, batch=self._training_batch())
            for i in range(self._n_train_repeat)
        ]

        diagnostics = tree.map_structure(
            lambda *d: tf.reduce_mean(d).numpy(), *diagnostics)

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat

        return diagnostics

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    def _init_training(self):
        pass

    @property
    def tf_saveables(self):
        return {}

    def __getstate__(self):
        state = {
            '_epoch_length': self._epoch_length,
            '_epoch': (
                    self._epoch + int(self._timestep >= self._epoch_length)),
            '_timestep': self._timestep % self._epoch_length,
            '_num_train_steps': self._num_train_steps,
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            Qs_Target,
            Discret_Action_Num,
            policy_lr=3e-4,
            Q_lr=3e-4,
            alpha_lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,

            save_full_state=False,
            Q_targets=None,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            Discret_Action_Num: 离散动作个数
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs

        self._Q_targets = Qs_Target


        self._policy_lr = policy_lr
        self._Q_lr = Q_lr
        self._alpha_lr = alpha_lr

        self._reward_scale = reward_scale
        self._target_entropy = Discret_Action_Num

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval

        self._save_full_state = save_full_state

        self._Q_optimizers = tuple(
            tf.optimizers.Adam(
                learning_rate=self._Q_lr,
                name=f'Q_{i}_optimizer'
            ) for i, Q in enumerate(self._Qs))

        self._policy_optimizer = tf.optimizers.Adam(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        self._log_alpha = tf.Variable(0.0)
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)

        self._alpha_optimizer = tf.optimizers.Adam(
            self._alpha_lr, name='alpha_optimizer')

    def _compute_Q_targets(self, batch):
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']


        entropy_scale = tf.convert_to_tensor(self._alpha)
        reward_scale = tf.convert_to_tensor(self._reward_scale)
        discount = tf.convert_to_tensor(self._discount)

        next_actions, next_log_pis = self._policy.actions_and_log_probs(
            next_observations)
        next_actions = next_actions.numpy()
        # 注意这里的next_actions 是tensor类型数据
        # print("next_observations:{}".format(next_observations))
        # print("next_actions:{},type:{}".format(next_actions,type(next_actions)))
        next_Qs_values = tuple(
            Q.values(next_observations, next_actions) for Q in self._Q_targets)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)

        Q_targets = compute_Q_targets(
            next_Q_values,
            next_log_pis,
            rewards,
            terminals,
            discount,
            entropy_scale,
            reward_scale)

        return tf.stop_gradient(Q_targets)

    def _update_critic(self, batch):
        """Update the Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_targets = self._compute_Q_targets(batch)

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q.values(observations, actions)
                Q_losses = 0.5 * (
                    tf.losses.MSE(y_true=Q_targets, y_pred=Q_values))
                Q_loss = tf.nn.compute_average_loss(Q_losses)

            gradients = tape.gradient(Q_loss, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    def _update_actor(self, batch):
        """Update the policy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        observations = batch['observations']
        entropy_scale = tf.convert_to_tensor(self._alpha)

        with tf.GradientTape() as tape:
            actions, log_pis = self._policy.actions_and_log_probs(observations)
            actions = actions.numpy()

            Qs_targets = tuple(
                Q.values(observations, actions) for Q in self._Qs)
            Q_targets = tf.reduce_mean(Qs_targets, axis=0)
            policy_losses = entropy_scale * log_pis - Q_targets
            policy_loss = tf.nn.compute_average_loss(policy_losses)

        policy_gradients = tape.gradient(
            policy_loss, self._policy.trainable_variables)

        self._policy_optimizer.apply_gradients(zip(
            policy_gradients, self._policy.trainable_variables))

        return policy_losses

    def _update_alpha(self, batch):
        if not isinstance(self._target_entropy, Number):
            return 0.0

        observations = batch['observations']
        # print("observations:{}".format(observations))
        actions, log_pis = self._policy.actions_and_log_probs(observations)
        # print("log_pis:{}".format(log_pis))

        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                    self._alpha * tf.stop_gradient(log_pis + self._target_entropy))
            # NOTE(hartikainen): It's important that we take the average here,
            # otherwise we end up effectively having `batch_size` times too
            # large learning rate.
            # print("alpha_losses:{}".format(alpha_losses))
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(
            alpha_gradients, [self._log_alpha]))

        return alpha_losses

    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    def _do_updates(self, batch):
        """Runs the update operations for policy, Q, and alpha."""
        Qs_values, Qs_losses = self._update_critic(batch)
        policy_losses = self._update_actor(batch)
        alpha_losses = self._update_alpha(batch)

        diagnostics = OrderedDict((
            ('Q_value-mean', tf.reduce_mean(Qs_values)),
            ('Q_loss-mean', tf.reduce_mean(Qs_losses)),
            ('policy_loss-mean', tf.reduce_mean(policy_losses)),
            ('alpha', tf.convert_to_tensor(self._alpha)),
            ('alpha_loss-mean', tf.reduce_mean(alpha_losses)),
        ))
        return diagnostics

    def _do_training(self, iteration, batch):
        training_diagnostics = self._do_updates(batch)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target(tau=tf.constant(self._tau))

        return training_diagnostics

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as an ordered dictionary.

        """
        diagnostics = OrderedDict((
            ('alpha', self._alpha.numpy()),
            ('policy', self._policy.get_diagnostics_np(batch['observations'])),
        ))

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_alpha': self._alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables

