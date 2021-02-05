# -*- coding: utf-8 -*-
"""
Tensor component analysis.

Inspired by Williams paper

Mostly developed during ANDA2019

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import neo
import quantities as pq
from elephant import kernels
from elephant.statistics import instantaneous_rate
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac


__all__ = [
    "get_instantaneous_rate_tensor",
    "get_sorted_unique_unit_names",
    "get_spiketrains_according_to_targdict_list",
    "normalize_instantaneous_rate_tensor"
]


def get_instantaneous_rate_tensor(block,
                                  sampling_period=10*pq.ms,
                                  kernel='gaussian',
                                  sigma=20*pq.ms,
                                  targdict_list=None,
                                  return_unit_names=False):

    trials = block.segments
    reference_trial_duration = trials[0].t_stop

    unit_names, unit_names_sorting = \
        get_sorted_unique_unit_names(block, targdict_list)
    n_units = len(unit_names)
    n_trials = len(trials)
    print(f'Creating instantaneous rate tensor for',
          f'{n_units} units, {n_trials} trials',
          f'For the instantaneous rate estimation, the spikes are',
          f'convolved with a {kernel} kernel with {sigma} sigma.')

    # by rescaling to pq.ms the sampling period is indirectly set to 1*pq.ms
    n_time_bins = int(
        reference_trial_duration.rescale(pq.ms)/sampling_period.rescale(pq.ms)
    )

    # initialize instantaneous_rate_tensor
    instantaneous_rate_tensor = np.zeros((n_units, n_trials, n_time_bins))

    # initialize kernel instance
    if kernel == 'gaussian':
        kernel_instance = kernels.GaussianKernel(sigma=sigma)
    elif kernel == 'triangular':
        kernel_instance = kernels.TriangularKernel(sigma=sigma)
    elif kernel == 'rectangular':
        kernel_instance = kernels.RectangularKernel(sigma=sigma)

    # loop over all trials
    for i, trial in enumerate(trials):
        instantaneous_rates = []

        if targdict_list:
            sts = get_spiketrains_according_to_targdict_list(trial,
                                                             targdict_list)
        else:
            sts = trial.spiketrains

        for st in sts:
            instantaneous_rates.append(
                instantaneous_rate(st,
                                   sampling_period=sampling_period,
                                   kernel=kernel_instance))
        instantaneous_rate_tensor[:, i, :] = \
            np.squeeze(np.array(instantaneous_rates))

    times = instantaneous_rates[0].times
    if return_unit_names:
        return instantaneous_rate_tensor[unit_names_sorting, :, :], unit_names, times
    return instantaneous_rate_tensor


def get_sorted_unique_unit_names(block,
                                 targdict_list=None):
    # obtain unit names
    unit_names = []
    first_trial = block.segments[0]
    if targdict_list:
        sts = get_spiketrains_according_to_targdict_list(
            first_trial,
            targdict_list)
    else:
        sts = first_trial.spiketrains

    for st in sts:
        # channel_id is integer, unit_id in channel is decimal
        implantation_site = st.annotations['implantation_site']
        if implantation_site == 'M1/PMd':
            implantation_site = 'M1'

        unique_unit_name = \
            implantation_site + '_' + str(st.annotations['channel_id'] +
                                          st.annotations['unit_id']/10)
        unit_names.append(unique_unit_name)

    unit_names_sorting = np.argsort(unit_names)
    unit_names = np.array(unit_names)[unit_names_sorting]
    return unit_names, unit_names_sorting


def get_spiketrains_according_to_targdict_list(segment, targdict_list):
    sts = []
    for targdict in targdict_list:
        st = segment.filter(objects=neo.SpikeTrain,
                            targdict=targdict)
        # TODO check if append or extend is correct
        sts.extend(st)
    return sts

# TODO implement soft-max normalization


def normalize_instantaneous_rate_tensor(instantaneous_rate_tensor):

    n_units = instantaneous_rate_tensor.shape[0]
    n_trials = instantaneous_rate_tensor.shape[1]
    n_time_bins = instantaneous_rate_tensor.shape[2]

    print("Normalizing firing rate of all units on each trial")
    rate_tensor_flat = instantaneous_rate_tensor.reshape(n_units, -1)
    rate_tensor_flat = (rate_tensor_flat.T / np.max(
        rate_tensor_flat, axis=-1)).T
    instantaneous_rate_tensor = rate_tensor_flat.reshape(
        n_units, n_trials, n_time_bins)

    return instantaneous_rate_tensor


def convert_to_tensorly_and_decompose(instantaneous_rate_tensor,
                                      n_factors,
                                      verbose=True,
                                      nonnegative=True,
                                      normalize_factors=True):

    tensorly_R = tl.tensor(instantaneous_rate_tensor)
    
    if nonnegative:
        CPtensor = non_negative_parafac(tensorly_R,
                                    rank=n_factors,
                                    verbose=verbose,
                                    normalize_factors=normalize_factors)
    
    else: 
        CPtensor = parafac(tensorly_R,
                                    rank=n_factors,
                                    verbose=verbose,
                                    normalize_factors=normalize_factors)
    return CPtensor