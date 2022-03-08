#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:variant_spec.py
@time:2021/11/11
"""

from copy import deepcopy

from ray import tune
import numpy as np
import collections
import os
M = 256
NUM_COUPLING_LAYERS = 2

def get_variant_spec(args):
    # 这三个参数唯一指定了gym环境
    universe, domain, task = args.universe, args.domain, args.task

    variant_spec = get_variant_spec_image(
        universe, domain, task, args.policy, args.algorithm)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec

def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    return variant_spec


ALGORITHM_PARAMS_BASE = {
    'config': {
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_kwargs': {},
        'eval_n_episodes': 1,
        'num_warmup_samples': tune.sample_from(lambda spec: (
                10 * (spec.get('config', spec)
        ['sampler_params']
        ['config']
        ['max_path_length'])
        )),
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'class_name': 'SAC',
        'config': {
            'policy_lr': 3e-4,
            'Q_lr': 3e-4,
            'alpha_lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',

            'discount': 0.99,
            'reward_scale': 1.0,
        },
    },
    'SQL': {
        'class_name': 'SQL',
        'config': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'discount': 0.99,
            'tau': 5e-3,
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        },
    },
}


def deep_update(d, *us):
    d = d.copy()

    for u in us:
        u = u.copy()
        for k, v in u.items():
            d[k] = (
                deep_update(d.get(k, {}), v)
                if isinstance(v, collections.Mapping)
                else v)

    return d


PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))

def get_git_rev(path=PROJECT_PATH, search_parent_directories=True):
    try:
        import git
    except ImportError:
        print(
            "Warning: gitpython not installed."
            " Unable to log git rev."
            " Run `pip install gitpython` if you want git revs to be logged.")
        return None

    try:
        repo = git.Repo(
            path, search_parent_directories=search_parent_directories)
        if repo.head.is_detached:
            git_rev = repo.head.object.name_rev
        else:
            git_rev = repo.active_branch.commit.name_rev
    except git.InvalidGitRepositoryError:
        git_rev = None

    return git_rev

def get_host_name():
    try:
        import socket
        return socket.gethostname()
    except Exception as e:
        print("Failed to get host name!")
        return None

def get_checkpoint_frequency(spec):
    num_checkpoints = 10
    config = spec.get('config', spec)
    checkpoint_frequency = (
                               config
                               ['algorithm_params']
                               ['config']
                               ['n_epochs']
                           ) // num_checkpoints

    return checkpoint_frequency

def get_variant_spec_base(universe, domain, task, policy, algorithm):
    # 算法参数需要自己处理下
    algorithm_params_environment = {
        'config': {
            'n_epochs': int(24),
            'epoch_length': 2016,
            'min_pool_size': 10000,
            'batch_size': 256,
        }
    }
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {}),
        algorithm_params_environment,
    )
    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        # 'policy_params': tune.sample_from(get_policy_params),
        'policy_params': {
            'class_name': 'FeedforwardGaussianPolicy',
            'config': {
                'hidden_layer_sizes': (M, M),
                'squash': True,
                'observation_keys': None,
                'preprocessors': None,
            },
        },
        'exploration_policy_params': {
            'class_name': 'ContinuousUniformPolicy',
            'config': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['config']
                        .get('observation_keys')
                ))
            },
        },
        'Q_params': {
            'class_name': 'double_feedforward_Q_function',
            'config': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'preprocessors': None,
            },
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'class_name': 'SimpleReplayPool',
            'config': {
                'max_size': tune.sample_from(lambda spec: (
                    min(int(1e6),
                        spec.get('config', spec)
                        ['algorithm_params']
                        ['config']
                        ['n_epochs']
                        * spec.get('config', spec)
                        ['algorithm_params']
                        ['config']
                        ['epoch_length'])
                )),
            },
        },
        'sampler_params': {
            'class_name': 'SimpleSampler',
            'config': {
                'max_path_length': 2016,
            }
        },
        'run_params': {
            'host_name': get_host_name(),
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec
