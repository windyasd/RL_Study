#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:practice.py
@time:2020/10/25
"""


if __name__ == '__main__':
    import numpy as np

    from grid2op import make
    from grid2op.Agent import RandomAgent

    max_iter = 10  # to make computation much faster we will only consider 50 time steps instead of 287
    train_iter = 10
    env_name = "rte_case14_realistic"
    # create an environment
    env = make(env_name)
    # don't forget to set "test=False" (or remove it, as False is the default value) for "real" training

    # import the train function and train your agent
    from l2rpn_baselines.DoubleDuelingDQN import train

    agent_name = "test_agent"
    save_path = "saved_agent_DDDQN_{}".format(train_iter)
    train(env,
          name=agent_name,
          iterations=train_iter,
          save_path=save_path,
          load_path=None,  # put something else if you want to reload an agent instead of creating a new one
          logs_path="tf_logs_DDDQN")