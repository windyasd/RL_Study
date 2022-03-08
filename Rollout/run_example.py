#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:run_example.py
@time:2021/11/11
"""
import importlib
import ray
from ray import tune
import os

import tensorflow as tf
import os
import random

import tensorflow as tf
import numpy as np
import tree
import copy
import grid2op
from grid2op.Reward import BaseReward, RedispReward, L2RPNSandBoxScore
import numpy as np
from grid2op.Parameters import Parameters
from Sampler import SimpleReplayPool,SimpleSampler
from ExperimentRunner import ExperimentRunner
from variant_spec import get_variant_spec
from get_parser import get_parser
from experiment_kwargs import generate_experiment_kwargs
from q_value_nwtwork import double_feedforward_Q_function
from discret_policy import GaussianPolicy
from SAC_Algorithm import SAC


def run_example_local(example_module_name, example_argv, local_mode=False):
    """Run example locally, potentially parallelizing across cpus/gpus."""
    # example_module_name:'examples.development'; --algorithm SAC  --universe gym  --domain HalfCheetah
    # --task v3  --exp-name my-sac-experiment-1 --checkpoint-frequency 1000  # Save the checkpoint to resume training later
    # 添加了argument parser, 并解析了相关参数
    example_args = get_parser().parse_args(example_argv)
    variant_spec = get_variant_spec(example_args)
    trainable_class = ExperimentRunner

    experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)


    ray.init(
        num_cpus=example_args.cpus,
        num_gpus=example_args.gpus,
        resources=example_args.resources or {},
        local_mode=local_mode,
        include_dashboard=example_args.include_dashboard,
        _temp_dir=example_args.temp_dir)

    tune.run(
        trainable_class,
        **experiment_kwargs,
        server_port=example_args.server_port,
        fail_fast=example_args.fail_fast,
        scheduler=None,
        reuse_actors=True)

def get_environment_from_params():
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

def run_grid2op(example_argv):
    Hidden_Layer_Sizes = (100,100)
    example_args = get_parser().parse_args(example_argv)
    variant_spec = get_variant_spec(example_args)
    variant = generate_experiment_kwargs(variant_spec, example_args)
    print("variant:{}".format(variant))
    environment_params = variant['config']['environment_params']
    training_environment = get_environment_from_params()
    evaluation_environment = get_environment_from_params()

    obs = training_environment.get_obs()
    observation = np.concatenate(([obs.rho.max()],[obs.rho.min()]))
    action = [0]
    # print("observation:{},action:{}".format(observation,action))
    Input_Demo_ = np.concatenate((observation,action))
    Input_Demo = Input_Demo_.reshape((1,len(Input_Demo_)))

    Qs = double_feedforward_Q_function(input_demo=Input_Demo,
                                       hidden_layer_sizes=Hidden_Layer_Sizes,
                                       name = 'Double_Q_Value_Function')

    Qs_Target = double_feedforward_Q_function(input_demo=Input_Demo,
                                       hidden_layer_sizes=Hidden_Layer_Sizes,
                                       name = 'Target_Double_Q_Value_Function')



    Action_Num = len(np.load('../Tutor/Single_Sub_Structure.npy',allow_pickle=True)) \
                 + len(np.load('../Tutor/Multiple_Sub_Structure_Last.npy',allow_pickle=True))+ 3
    Input_Demo = observation.reshape((1,len(observation)))
    policy = GaussianPolicy(input_demo=Input_Demo,output_shape=Action_Num,
                            hidden_layer_sizes=Hidden_Layer_Sizes)

    # 这个需要重新指定,path_length 和 环境

    replay_pool = SimpleReplayPool(environment=training_environment, max_size=10000)

    path_length = 4
    sampler = SimpleSampler(
        environment=training_environment,
        policy=policy,
        pool=replay_pool,
        max_path_length=path_length)

    for i in range(5):
        sampler.sample()

    # print('replay_pool.data:{}'.format(replay_pool.data))

    algorithm = SAC(training_environment = training_environment,
                         evaluation_environment = evaluation_environment,
                         policy = policy,
                         Qs = Qs,
                         Qs_Target=Qs_Target,
                         Discret_Action_Num = Action_Num,
                         sampler = sampler,
                         pool = replay_pool
                         )


if __name__ == '__main__':
    example_argv = {}
    Hidden_Layer_Sizes = (100,100)
    example_args = get_parser().parse_args(example_argv)
    variant_spec = get_variant_spec(example_args)
    variant = generate_experiment_kwargs(variant_spec, example_args)
    print("variant:{}".format(variant))
    print("variant['config']:{}".format(variant['config']))
    environment_params = variant['config']['environment_params']
    training_environment = get_environment_from_params()
    evaluation_environment = get_environment_from_params()

    obs = training_environment.get_obs()
    observation = np.concatenate(([obs.rho.max()],[obs.rho.min()]))
    action = [0]
    # print("observation:{},action:{}".format(observation,action))
    Input_Demo_ = np.concatenate((observation,action))
    Input_Demo = Input_Demo_.reshape((1,len(Input_Demo_)))

    Qs = double_feedforward_Q_function(input_demo=Input_Demo,
                                       hidden_layer_sizes=Hidden_Layer_Sizes,
                                       name = 'Double_Q_Value_Function')

    Qs_Target = double_feedforward_Q_function(input_demo=Input_Demo,
                                              hidden_layer_sizes=Hidden_Layer_Sizes,
                                              name = 'Target_Double_Q_Value_Function')


    Action_Num = len(np.load('../Tutor/Single_Sub_Structure.npy',allow_pickle=True)) \
                 + len(np.load('../Tutor/Multiple_Sub_Structure_Last.npy',allow_pickle=True))+ 3
    Input_Demo = observation.reshape((1,len(observation)))
    policy = GaussianPolicy(input_demo=Input_Demo,output_shape=Action_Num,
                            hidden_layer_sizes=Hidden_Layer_Sizes)

    # 这个需要重新指定,path_length 和 环境

    replay_pool = SimpleReplayPool(environment=training_environment, max_size=10000)

    path_length = 4
    sampler = SimpleSampler(
        environment=training_environment,
        policy=policy,
        pool=replay_pool,
        max_path_length=path_length)

    # for i in range(5):
    #     sampler.sample()

    # print('replay_pool.data:{}'.format(replay_pool.data))

    algorithm = SAC(training_environment = training_environment,
                    evaluation_environment = evaluation_environment,
                    policy = policy,
                    Qs = Qs,
                    Qs_Target=Qs_Target,
                    Discret_Action_Num = Action_Num,
                    sampler = sampler,
                    pool = replay_pool,
                    min_pool_size=20,
                    batch_size=5,
                    n_epochs=10,
                    epoch_length=2016,
                    train_every_n_steps=3,
                    n_train_repeat=3,
                    target_update_interval=100
                    )
    algorithm.train()
