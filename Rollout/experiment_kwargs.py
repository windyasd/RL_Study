#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:experiment_kwargs.py
@time:2021/11/11
"""
import os
import datetime

def _normalize_trial_resources(resources, cpu, gpu, extra_cpu, extra_gpu):
    if resources is None:
        resources = {}

    if cpu is not None:
        resources['cpu'] = cpu

    if gpu is not None:
        resources['gpu'] = gpu

    if extra_cpu is not None:
        resources['extra_cpu'] = extra_cpu

    if extra_gpu is not None:
        resources['extra_gpu'] = extra_gpu

    return resources

def add_command_line_args_to_variant_spec(variant_spec, command_line_args):
    variant_spec['run_params'].update({
        'checkpoint_frequency': (
            command_line_args.checkpoint_frequency
            if command_line_args.checkpoint_frequency is not None
            else variant_spec['run_params'].get('checkpoint_frequency', 0)
        ),
        'checkpoint_at_end': (
            command_line_args.checkpoint_at_end
            if command_line_args.checkpoint_at_end is not None
            else variant_spec['run_params'].get('checkpoint_at_end', True)
        ),
    })

    if (command_line_args.mode == 'debug'
            and ('run_eagerly' not in command_line_args
                 or command_line_args.run_eagerly is None)):
        variant_spec['run_params']['run_eagerly'] = True
    elif 'run_eagerly' in command_line_args:
        variant_spec['run_params']['run_eagerly'] = (
            command_line_args.run_eagerly)

    variant_spec['restore'] = command_line_args.restore

    return variant_spec

def datetimestamp(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider, dtd=datetime_divider))

# experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)
def generate_experiment_kwargs(variant_spec, command_line_args):
    local_dir = command_line_args.local_dir
    if command_line_args.mode == 'debug':
        local_dir = os.path.join(local_dir, 'debug')
    local_dir = os.path.join(
        local_dir,
        command_line_args.universe,
        command_line_args.domain,
        command_line_args.task)
    resources_per_trial = _normalize_trial_resources(
        command_line_args.resources_per_trial,
        command_line_args.trial_cpus,
        command_line_args.trial_gpus,
        command_line_args.trial_extra_cpus,
        command_line_args.trial_extra_gpus)
    upload_dir = (
        os.path.join(
            command_line_args.upload_dir,
            command_line_args.universe,
            command_line_args.domain,
            command_line_args.task)
        if command_line_args.upload_dir
        else None)

    datetime_prefix = datetimestamp()
    experiment_id = '-'.join((datetime_prefix, command_line_args.exp_name))

    variant_spec = add_command_line_args_to_variant_spec(
        variant_spec, command_line_args)

    if command_line_args.video_save_frequency is not None:
        assert 'algorithm_params' in variant_spec
        variant_spec['algorithm_params']['config']['video_save_frequency'] = (
            command_line_args.video_save_frequency)

    def create_trial_name_creator(trial_name_template=None):
        if not trial_name_template:
            return None

        def trial_name_creator(trial):
            return trial_name_template.format(trial=trial)

        return trial_name_creator

    experiment_kwargs = {
        'name': experiment_id,
        'resources_per_trial': resources_per_trial,
        'config': variant_spec,
        'local_dir': local_dir,
        'num_samples': command_line_args.num_samples,
        'upload_dir': upload_dir,
        'checkpoint_freq': (
            variant_spec['run_params']['checkpoint_frequency']),
        'checkpoint_at_end': (
            variant_spec['run_params']['checkpoint_at_end']),
        'max_failures': command_line_args.max_failures,
        'trial_name_creator': create_trial_name_creator(
            command_line_args.trial_name_template),
        'restore': command_line_args.restore,  # Defaults to None
    }

    return experiment_kwargs



