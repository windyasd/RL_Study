#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:Sampler.py
@time:2021/11/11
"""
from collections import deque, OrderedDict
from itertools import islice
import numpy as np
import tree
from dataclasses import dataclass
from typing import Union, Callable
from numbers import Number
import gzip
import pickle

import numpy as np
import tensorflow as tf
import tree
import abc

import grid2op
from grid2op.Reward import BaseReward, RedispReward, L2RPNSandBoxScore
import numpy as np
from grid2op.Parameters import Parameters

# replay buffer设定

@dataclass
class Field:
    name: str
    dtype: Union[str, np.dtype, tf.DType]
    shape: Union[tuple, tf.TensorShape]
    initializer: Callable = np.zeros
    default_value: Number = 0.0

INDEX_FIELDS = {
    'episode_index_forwards': Field(
        name='episode_index_forwards',
        dtype='uint64',
        shape=(1, ),
        default_value=0,
    ),
    'episode_index_backwards': Field(
        name='episode_index_backwards',
        dtype='uint64',
        shape=(1, ),
        default_value=0,
    ),
}

class ReplayPool(object):
    """A class used to save and replay data."""

    @abc.abstractmethod
    def add_sample(self, sample):
        """Add a transition tuple."""
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """Clean up pool after episode termination."""
        pass

    @property
    @abc.abstractmethod
    def size(self, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def add_path(self, path):
        """Add a rollout to the replay pool."""
        pass

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """Return a random batch of size `batch_size`."""
        pass

class FlexibleReplayPool(ReplayPool):
    def __init__(self, max_size, fields):
        super(FlexibleReplayPool, self).__init__()

        max_size = int(max_size)
        self._max_size = max_size

        self.fields = {**fields, **INDEX_FIELDS}
        # print('fields:{}'.format(fields))
        self.data = tree.map_structure(self._initialize_field, self.fields)
        # print("初始时，self.data：{}".format(self.data))

        self._pointer = 0
        self._size = 0
        self._samples_since_save = 0

    @property
    def size(self):
        return self._size

    def _initialize_field(self, field):
        # 这里的关键是field.shape，指的是要存储数据的维度
        field_shape = (self._max_size, *field.shape)
        # print('field.name:{},field_shape:{}'.format(field.name,field_shape))
        # np.zeros()
        field_values = field.initializer(
            field_shape, dtype=field.dtype)

        return field_values

    def _advance(self, count=1):
        """Handles bookkeeping after adding samples to the pool.

        * Moves the pointer (`self._pointer`)
        * Updates the size (`self._size`)
        * Fixes the `episode_index_backwards` field, which might have become
          out of date when the pool is full and we start overriding old
          samples.
        """
        self._pointer = (self._pointer + count) % self._max_size
        self._size = min(self._size + count, self._max_size)

        if self.data['episode_index_forwards'][self._pointer] != 0:
            episode_tail_length = int(self.data[
                                          'episode_index_backwards'
                                      ][self._pointer, 0] + 1)
            self.data[
                'episode_index_forwards'
            ][np.arange(
                self._pointer, self._pointer + episode_tail_length
            ) % self._max_size] = np.arange(episode_tail_length)[..., None]

        self._samples_since_save += count

    def add_sample(self, sample):
        samples = tree.map_structure(lambda x: x[..., np.newaxis], sample)
        self.add_samples(samples)

    def add_samples(self, samples):
        num_samples = tree.flatten(samples)[0].shape[0]

        assert (('episode_index_forwards' in samples.keys())
                is ('episode_index_backwards' in samples.keys()))
        if 'episode_index_forwards' not in samples.keys():
            samples['episode_index_forwards'] = np.full(
                (num_samples, *self.fields['episode_index_forwards'].shape),
                self.fields['episode_index_forwards'].default_value,
                dtype=self.fields['episode_index_forwards'].dtype)
            samples['episode_index_backwards'] = np.full(
                (num_samples, *self.fields['episode_index_backwards'].shape),
                self.fields['episode_index_backwards'].default_value,
                dtype=self.fields['episode_index_backwards'].dtype)

        index = np.arange(
            self._pointer, self._pointer + num_samples) % self._max_size

        def add_sample(self,data, new_values, field):
            # print("data:{}".format(data))
            # print("new_values:{}".format(new_values))
            data[index] = new_values
            # print("data:{}".format(data))

        tree.map_structure_with_path(add_sample, self.data, samples, self.fields)

        self._advance(num_samples)

    def add_path(self, path):
        # 给数据添加了 'episode_index_forwards'和'episode_index_backwards'字段，一个是【0，1，2，path_length】，另一个是倒序
        path = path.copy()
        # print("path:{}".format(path))
        path_length = tree.flatten(path)[0].shape[0]
        path.update({
            'episode_index_forwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_forwards'].dtype
            )[..., np.newaxis],
            'episode_index_backwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_backwards'].dtype
            )[::-1, np.newaxis],
        })

        return self.add_samples(path)

    def random_indices(self, batch_size):
        if self._size == 0: return np.arange(0, 0)
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)

    def random_sequence_batch(self, batch_size, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.sequence_batch_by_indices(random_indices, **kwargs)

    def last_n_batch(self, last_n, field_name_filter=None, **kwargs):
        last_n_indices = np.arange(
            self._pointer - min(self.size, int(last_n)), self._pointer,
            dtype=int
        ) % self._max_size

        return self.batch_by_indices(
            last_n_indices, field_name_filter=field_name_filter, **kwargs)

    def last_n_sequence_batch(self, last_n, **kwargs):
        last_n_indices = np.arange(
            self._pointer - min(self.size, int(last_n)), self._pointer,
            dtype=int
        ) % self._max_size

        return self.sequence_batch_by_indices(last_n_indices, **kwargs)

    def filter_fields(self, field_names, field_name_filter):
        if isinstance(field_name_filter, str):
            field_name_filter = [field_name_filter]

        if isinstance(field_name_filter, (list, tuple)):
            field_name_list = field_name_filter

            def filter_fn(field_name):
                return field_name in field_name_list

        else:
            filter_fn = field_name_filter

        filtered_field_names = [
            field_name for field_name in field_names
            if filter_fn(field_name)
        ]

        return filtered_field_names

    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         validate_index=True):
        if validate_index and np.any(self.size <= indices % self._max_size):
            raise ValueError(
                "Tried to retrieve batch with indices greater than current"
                " size")

        if field_name_filter is not None:
            raise NotImplementedError("TODO(hartikainen)")

        # print("self.data:{}".format(self.data))

        batch = tree.map_structure(
            lambda field: field[indices % self._max_size], self.data)
        return batch

    def sequence_batch_by_indices(self,
                                  indices,
                                  sequence_length,
                                  field_name_filter=None):
        if np.any(self.size <= indices % self._max_size):
            raise ValueError(
                "Tried to retrieve batch with indices greater than current"
                " size")
        if indices.size < 1:
            return self.batch_by_indices(indices)

        sequence_indices = (
                indices[:, None] + np.arange(sequence_length)[None])
        sequence_batch = self.batch_by_indices(
            sequence_indices, validate_index=False)

        if 'mask' in sequence_batch:
            raise ValueError(
                "sequence_batch_by_indices adds a field 'mask' into the batch."
                " There already exists a 'mask' field in the batch. Please"
                " remove it before using sequence_batch. TODO(hartikainen):"
                " Allow mask name to be configured.")

        forward_diffs_0 = np.diff(
            sequence_batch['episode_index_forwards'].astype(np.int64), axis=1)
        forward_diffs_1 = np.pad(
            forward_diffs_0, ([0, 0], [0, 1], [0, 0]),
            mode='constant',
            constant_values=-1)
        cut_and_pad_sample_indices = (
                np.argmax(forward_diffs_1[:, ::1, :] < 1, axis=1)
                + 1)[..., 0]

        sequence_batch['mask'] = np.where(
            np.arange(sequence_length)[None, ...]
            < cut_and_pad_sample_indices[..., None],
            True,
            False)

        return sequence_batch

    def save_latest_experience(self, pickle_path):
        latest_samples = self.last_n_batch(self._samples_since_save)

        with gzip.open(pickle_path, 'wb') as f:
            pickle.dump(latest_samples, f)

        self._samples_since_save = 0

    def load_experience(self, experience_path):
        with gzip.open(experience_path, 'rb') as f:
            latest_samples = pickle.load(f)

        num_samples = tree.flatten(latest_samples)[0].shape[0]

        def assert_shape(data):
            assert data.shape[0] == num_samples, data.shape

        tree.map_structure(assert_shape, latest_samples)

        self.add_samples(latest_samples)
        self._samples_since_save = 0

# 这里要指定初始observation 和action的格式
class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self,
                 environment,
                 *args,
                 extra_fields=None,
                 **kwargs):
        extra_fields = extra_fields or {}
        ################## 重新指定observation_space #################
        obs = environment.get_obs()
        # 处理后的observation
        observation_space = np.concatenate(([obs.rho.max()],[obs.rho.min()]))
        # 动作编号
        action_space = np.array(0)

        ############################################################
        # observation_space = environment.observation_space
        # action_space = environment.action_space

        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space

        fields = {
            'observations':Field(
                name='observations',
                dtype=observation_space.dtype,
                shape=observation_space.shape),
            'next_observations':Field(
                name='next_observations',
                dtype=observation_space.dtype,
                shape=observation_space.shape),
            'actions': Field(
                name='actions',
                dtype=action_space.dtype,
                shape=action_space.shape),
            'rewards': Field(
                name='rewards',
                dtype='float32',
                shape=(1, )),
            # terminals[i] = a terminal was received at time i
            'terminals': Field(
                name='terminals',
                dtype='bool',
                shape=(1, )),
            **extra_fields
        }

        super(SimpleReplayPool, self).__init__(
            *args, fields=fields, **kwargs)

# Trajectory 采集器

class BaseSampler(object):
    def __init__(self,
                 max_path_length,
                 environment=None,
                 policy=None,
                 pool=None,
                 store_last_n_paths=10):
        self._max_path_length = max_path_length
        self._store_last_n_paths = store_last_n_paths
        self._last_n_paths = deque(maxlen=store_last_n_paths)

        self.environment = environment
        self.policy = policy
        self.pool = pool

    def initialize(self, environment, policy, pool):
        self.environment = environment
        self.policy = policy
        self.pool = pool

    def reset(self):
        pass

    def set_policy(self, policy):
        self.policy = policy

    def clear_last_n_paths(self):
        self._last_n_paths.clear()

    def get_last_n_paths(self, n=None):
        if n is None:
            n = self._store_last_n_paths

        last_n_paths = tuple(islice(self._last_n_paths, None, n))

        return last_n_paths

    def sample(self):
        raise NotImplementedError

    def terminate(self):
        self.environment.close()

    def get_diagnostics(self):
        diagnostics = OrderedDict({'pool-size': self.pool.size})
        return diagnostics

    def __getstate__(self):
        state = {
            key: value for key, value in self.__dict__.items()
            if key not in (
                'environment',
                'policy',
                'pool',
                '_last_n_paths',
                '_current_observation',
                '_current_path',
                '_is_first_step',
            )
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.environment = None
        self.policy = None
        self.pool = None
        # TODO(hartikainen): Maybe try restoring these from the pool?
        self._last_n_paths = deque(maxlen=self._store_last_n_paths)


def topo_change_sub_action_with_reconnect(topo_old, topo_new, env):

    res = []
    # 需要改变的topo_vect位置
    change_location = []
    for index, value in enumerate(topo_old):
        if value != topo_new[index] and topo_new[index]!=-1:
            change_location.append(index)
    # 需要改变的子站序列
    new_topo = np.full(shape=len(topo_old), fill_value=0, dtype=int)
    new_topo[change_location] = 1

    action = env.action_space({})
    topo_sub_vect = action._topo_vect_to_sub
    topo_sub_vect_ = topo_sub_vect + 1

    sub_need_changed = []
    for i in set(np.multiply(np.array(new_topo), np.array(topo_sub_vect_))):
        if i > 0:
            sub_need_changed.append(i - 1)

    # 各子站相应的动作
    for sub_id in sub_need_changed:
        topo_vect_location = np.where(topo_sub_vect == sub_id)[0]
        topo_null = np.full(shape=len(topo_old), fill_value=0, dtype=int)
        for i in change_location:
            if i in topo_vect_location:
                topo_null[i] = topo_new[i]
        vect_single_sub = topo_null[topo_vect_location]
        res.append((sub_id, np.asarray(vect_single_sub)))
    return res
    # action = env.action_space({})
    # action.sub_set_bus = res
    # return action

def topo_change_to_initial_sub_action_vect(topo_old, topo_new, env):
    res = []
    # 需要改变的topo_vect位置
    change_location = []
    for index, value in enumerate(topo_old):
        if value != topo_new[index]:
            change_location.append(index)
    # 需要改变的子站序列
    new_topo = np.full(shape=len(topo_old), fill_value=0, dtype=int)
    new_topo[change_location] = 1

    action = env.action_space({})
    topo_sub_vect = action._topo_vect_to_sub
    topo_sub_vect_ = topo_sub_vect + 1

    sub_need_changed = []
    for i in set(np.multiply(np.array(new_topo), np.array(topo_sub_vect_))):
        if i > 0:
            sub_need_changed.append(i - 1)

    # 各子站相应的动作
    for sub_id in sub_need_changed:
        topo_vect_location = np.where(topo_sub_vect == sub_id)[0]
        topo_null = np.full(shape=len(topo_old), fill_value=0, dtype=int)
        for i in change_location:
            if i in topo_vect_location:
                topo_null[i] = topo_new[i]
        vect_single_sub = topo_null[topo_vect_location]
        res.append((sub_id, np.asarray(vect_single_sub)))
    return res

def reconnect_array(obs):
    """
    ⚠️目前每次只能改变一条线路状态
        返回重连线路状态向量
        :param obs:
        :return:
        """
    res = []
    new_line_status_array = np.zeros_like(obs.rho)
    disconnected_lines = np.where(obs.line_status == False)[0]
    for line in disconnected_lines[::-1]:
        # print("第{}条线路，time_before_cooldown_line:{}".format(line, obs.time_before_cooldown_line[line]))
        if not obs.time_before_cooldown_line[line]:
            # this line is disconnected, and, it is not cooling down.
            line_to_reconnect = line
            new_line_status_array[line_to_reconnect] = 1
            res.append(new_line_status_array)
            new_line_status_array = np.zeros_like(obs.rho)
            # break  # reconnect the first one
    return res

def action_id_to_real_action(action_id,env):
    Single_Sub_Structure = np.load('../Tutor/Single_Sub_Structure.npy',allow_pickle=True)
    Double_Sub_Structure = np.load('../Tutor/Multiple_Sub_Structure_Last.npy',allow_pickle=True)

    obs = env.get_obs()
    # 自己生成 topo_vect_initial
    topo_vect_initial = []
    for _ in range(len(obs.topo_vect)):
        topo_vect_initial.append(1)


    # 单子站结构动作
    if action_id<len(Single_Sub_Structure):
        # print("_______________  返回单子站动作  _______________")
        topo_structure = Single_Sub_Structure[action_id]
        obs = env.get_obs()
        sub_action_vect_handle = topo_change_sub_action_with_reconnect(obs.topo_vect, topo_structure, env)
        # print("动作涉及子站：{}".format(len(sub_action_vect_handle)))
        action = env.action_space({})
        action.sub_set_bus = sub_action_vect_handle
        if len(sub_action_vect_handle)>1:
            return action,sub_action_vect_handle
    # 双子站结构动作,返回两个动作
    elif action_id<len(Single_Sub_Structure)+len(Double_Sub_Structure):
        # print("_______________  返回双子站动作  _______________")
        topo_structure = Double_Sub_Structure[action_id-len(Single_Sub_Structure)]
        obs = env.get_obs()
        sub_action_vect_handle = topo_change_sub_action_with_reconnect(obs.topo_vect, topo_structure, env)
        action = env.action_space({})
        action.sub_set_bus = sub_action_vect_handle

        return action,sub_action_vect_handle

        # action_list = []
        #
        # for sub_action in sub_action_vect_handle:
        #     action = env.action_space({})
        #     action.sub_set_bus = sub_action
        #     action_list.append(action)
        # return action_list

    # None Action
    if action_id == len(Single_Sub_Structure)+len(Double_Sub_Structure):
        action = env.action_space({})
    # 恢复线路结构,返回多个动作
    if action_id == len(Single_Sub_Structure)+len(Double_Sub_Structure)+1:
        # print("_______________  返回恢复线路结构动作  _______________")
        obs = env.get_obs()
        sub_action_vect = topo_change_to_initial_sub_action_vect(obs.topo_vect, topo_vect_initial, env)
        action = env.action_space({})
        action.sub_set_bus = sub_action_vect
        if len(sub_action_vect)>1:
            return action,sub_action_vect

        # action_list = []
        # for sub_action in sub_action_vect:
        #     action = env.action_space({})
        #     action.sub_set_bus = sub_action
        #     action_list.append(action)
        # return action_list
    # 重连线路
    if action_id == len(Single_Sub_Structure)+len(Double_Sub_Structure)+2:
        obs = env.get_obs()
        reconnect_array_list = reconnect_array(obs)
        if len(reconnect_array_list)>0:
            rec_array = reconnect_array_list[0]
            action = env.action_space({})
            action.update({'set_line_status': rec_array.astype(int)})
        else:
            action = env.action_space({})

    if action_id>len(Single_Sub_Structure)+len(Double_Sub_Structure)+2:
        action = env.action_space({})
        # print("--------------------------  更正神经网络的action_length  --------------------------")

    return action,[]

# 使用时要处理下observation
class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._total_samples = 0

        self._is_first_step = True

    def reset(self):
        if self.policy is not None:
            self.policy.reset()

        self._path_length = 0
        self._path_return = 0
        self._current_path = []
        #################### 修改observation #####################
        obs = self.environment.reset()
        self._current_observation = np.concatenate(([obs.rho.max()],[obs.rho.min()]))
        #########################################
        # self._current_observation = self.environment.reset()

    @property
    def _policy_input(self):
        return self._current_observation

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': np.atleast_1d(reward),
            'terminals': np.atleast_1d(terminal),
            'next_observations': next_observation,
        }

        return processed_observation

    def sample(self):
        if self._is_first_step:
            self.reset()

        # action = self.policy.action(self._policy_input).numpy()
        ##########################  处理action #############################
        action = self.policy.action(self._current_observation)
        # print("action:{}".format(action))
        action_real,_ = action_id_to_real_action(action,self.environment)
        # print("action_real:{}".format(action_real))
        #######################################################

        next_obs, reward, terminal, info = self.environment.step(action_real)
        # print("next_obs:{},terminal:{}".format(next_obs,terminal))
        next_observation=np.concatenate(([next_obs.rho.max()],[next_obs.rho.min()]))
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
        )
        # print("processed_sample:{}".format(processed_sample))

        self._current_path.append(processed_sample)

        if terminal or self._path_length >= self._max_path_length:
            # print("self._current_path:{}".format(self._current_path))
            last_path = tree.map_structure(
                lambda *x: np.stack(x, axis=0), *self._current_path)

            self.pool.add_path({
                key: value
                for key, value in last_path.items()
                if key != 'infos'
            })

            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return
            self._n_episodes += 1

            self.pool.terminate_episode()

            self._is_first_step = True
            # Reset is done in the beginning of next episode, see above.

        else:
            self._current_observation = next_observation
            self._is_first_step = False

        return next_observation, reward, terminal, info

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics

class agent_policy():
    def __init__(self):
        pass
    def action(self,env):
        return 0
    def reset(self):
        pass

if __name__ == '__main__':
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

    path_length = 4
    pool = SimpleReplayPool(environment=env, max_size=path_length)
    agent = agent_policy()
    sampler = SimpleSampler(
        environment=env,
        policy=agent,
        pool=pool,
        max_path_length=path_length)

    for i in range(8):
        sampler.sample()