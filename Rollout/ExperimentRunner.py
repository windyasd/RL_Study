#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:ExperimentRunner.py
@time:2021/11/11
"""
import os
import copy
import glob
import pickle
import sys
import json

import tensorflow as tf
import tree
import ray
from ray import tune
import importlib
import ray
from ray import tune
import os
import datetime
import tensorflow as tf
import os
import random
from Sampler import SimpleReplayPool,SimpleSampler
import grid2op
from grid2op.Reward import BaseReward, RedispReward, L2RPNSandBoxScore
import numpy as np
from grid2op.Parameters import Parameters

import tensorflow as tf
import numpy as np
import tree
import copy
from q_value_nwtwork import double_feedforward_Q_function
from discret_policy import GaussianPolicy
from SAC_Algorithm import SAC

def set_seed(seed):
    seed %= 4294967294
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Using seed {seed}")

def set_gpu_memory_growth(growth):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, growth)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def get_environment_from_params(environment_params):
    other_rewards = {}
    other_rewards["tmp_score_codalab"] = L2RPNSandBoxScore
    input_dir = '../input_data_local'
    parameters = Parameters()
    parameters.HARD_OVERFLOW_THRESHOLD = 3.0
    parameters.MAX_SUB_CHANGED = 6
    parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 4
    parameters.MAX_LINE_STATUS_CHANGED = 100

    env = grid2op.make(input_dir, param=parameters,
                       reward_class=RedispReward,
                       other_rewards=other_rewards)
    env.seed(10)
    env.set_id(0)
    obs = env.reset()

    return env

class ExperimentRunner(tune.Trainable):
    def setup(self, variant):
        # Set the current working directory such that the local mode
        # logs into the correct place. This would not be needed on
        # local/cluster mode.
        if ray.worker._mode() == ray.worker.LOCAL_MODE:
            os.chdir(os.getcwd())

        set_seed(variant['run_params']['seed'])

        if variant['run_params'].get('run_eagerly', False):
            tf.config.experimental_run_functions_eagerly(True)

        self._variant = variant
        set_gpu_memory_growth(True)

        self.train_generator = None
        self._built = False

    def _build(self):
        variant = copy.deepcopy(self._variant)
        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        variant['Q_params']['config'].update({
            'input_shapes': (
                training_environment.observation_shape,
                training_environment.action_shape),
        })
        observation = training_environment.get_obs()
        action = [0]
        Input_Demo_ = np.concatenate((observation,action),axis=-1)
        Input_Demo = Input_Demo_.reshape((1,len(Input_Demo_)))

        Qs = double_feedforward_Q_function(input_demo=Input_Demo,
                                           hidden_layer_sizes=variant['Q_params']['config']['hidden_layer_sizes'],
                                           name = variant['Q_params']['class_name'])

        variant['policy_params']['config'].update({
            'action_range': (training_environment.action_space.low,
                             training_environment.action_space.high),
            'input_shapes': training_environment.observation_shape,
            'output_shape': training_environment.action_shape,
        })

        Action_Num = 360
        Input_Demo = observation.reshape((1,len(observation)))
        policy = GaussianPolicy(input_demo=Input_Demo,output_shape=Action_Num,
                                hidden_layer_sizes=variant['policy_params']['config']['hidden_layer_sizes'])


        variant['replay_pool_params']['config'].update({
            'environment': training_environment,
        })

        # 这个需要重新指定,path_length 和 环境
        path_length = 4
        replay_pool = SimpleReplayPool(environment=training_environment, max_size=path_length)

        variant['sampler_params']['config'].update({
            'environment': training_environment,
            'policy': policy,
            'pool': replay_pool,
        })

        sampler = SimpleSampler(
            environment=training_environment,
            policy=policy,
            pool=replay_pool,
            max_path_length=path_length)


        variant['algorithm_params']['config'].update({
            'training_environment': training_environment,
            'evaluation_environment': evaluation_environment,
            'policy': policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler
        })

        self.algorithm = SAC(training_environment = training_environment,
                             evaluation_environment = evaluation_environment,
                             policy = policy,
                             Qs = Qs,
                             Discret_Action_Num = Action_Num,
                             sampler = sampler,
                             pool = replay_pool
                            )

        self._built = True

    def step(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)

        return diagnostics

    @staticmethod
    def _pickle_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    @staticmethod
    def _algorithm_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'algorithm')

    @staticmethod
    def _replay_pool_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    @staticmethod
    def _sampler_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'sampler.pkl')

    @staticmethod
    def _policy_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'policy')

    def _save_replay_pool(self, checkpoint_dir):
        if not self._variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        replay_pool_save_path = self._replay_pool_save_path(checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_save_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        if not self._variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        experiment_root = os.path.dirname(current_checkpoint_dir)

        experience_paths = [
            self._replay_pool_save_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

    def _save_sampler(self, checkpoint_dir):
        with open(self._sampler_save_path(checkpoint_dir), 'wb') as f:
            pickle.dump(self.sampler, f)

    def _restore_sampler(self, checkpoint_dir):
        with open(self._sampler_save_path(checkpoint_dir), 'rb') as f:
            sampler = pickle.load(f)

        self.sampler.__setstate__(sampler.__getstate__())
        self.sampler.initialize(
            self.training_environment, self.policy, self.replay_pool)

    def _save_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.save_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('Q', *[str(x) for x in path]))),
                save_format='tf'),
            self.Qs)

    def _restore_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.load_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('Q', *[str(x) for x in path])))),
            self.Qs)

    def _save_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        self.policy.save(save_path)

    def _restore_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        status = self.policy.load_weights(save_path)
        status.assert_consumed().run_restore_ops()

    def _save_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)
        tf_checkpoint.save(file_prefix=f"{save_path}/checkpoint")

        state = self.algorithm.__getstate__()
        with open(os.path.join(save_path, "state.json"), 'w') as f:
            json.dump(state, f)

    def _restore_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        with open(os.path.join(save_path, "state.json"), 'r') as f:
            state = json.load(f)

        self.algorithm.__setstate__(state)

        # NOTE(hartikainen): We need to run one step on optimizers s.t. the
        # variables get initialized.
        # TODO(hartikainen): This should be done somewhere else.
        tree.map_structure(
            lambda Q_optimizer, Q: Q_optimizer.apply_gradients([
                (tf.zeros_like(variable), variable)
                for variable in Q.trainable_variables
            ]),
            tuple(self.algorithm._Q_optimizers),
            tuple(self.Qs),
        )

        self.algorithm._alpha_optimizer.apply_gradients([(
            tf.zeros_like(self.algorithm._log_alpha), self.algorithm._log_alpha
        )])
        self.algorithm._policy_optimizer.apply_gradients([
            (tf.zeros_like(variable), variable)
            for variable in self.policy.trainable_variables
        ])

        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)

        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            # os.path.split(f"{save_path}/checkpoint")[0])
            # f"{save_path}/checkpoint-xxx"))
            os.path.split(os.path.join(save_path, "checkpoint"))[0]))
        status.assert_consumed().run_restore_ops()

    def save_checkpoint(self, checkpoint_dir):
        """Implements the checkpoint save logic."""
        self._save_replay_pool(checkpoint_dir)
        self._save_sampler(checkpoint_dir)
        self._save_value_functions(checkpoint_dir)
        self._save_policy(checkpoint_dir)
        self._save_algorithm(checkpoint_dir)

        return os.path.join(checkpoint_dir, '')

    def load_checkpoint(self, checkpoint_dir):
        """Implements the checkpoint restore logic."""
        assert isinstance(checkpoint_dir, str), checkpoint_dir
        checkpoint_dir = checkpoint_dir.rstrip('/')

        self._build()

        self._restore_replay_pool(checkpoint_dir)
        self._restore_sampler(checkpoint_dir)
        self._restore_value_functions(checkpoint_dir)
        self._restore_policy(checkpoint_dir)
        self._restore_algorithm(checkpoint_dir)

        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True


