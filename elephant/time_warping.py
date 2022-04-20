# -*- coding: utf-8 -*-
"""
Time warping

- piecewise linear


Background
----------

- comparison to traditional trigger alignment
- challenge if trials of an experiment are not of same timing/duration


Functions overview
------------------

.. autosummary::
    :toctree: toctree/time_warping/

:copyright: Copyright 2015-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals
import warnings

import neo
from neo import utils
import numpy as np
import quantities as pq
import copy
import numba
from scipy.interpolate import interp1d

__all__ = [
    "warp_sequence_of_time_points",
    "warp_spiketrain_by_knots",
    "warp_list_of_spiketrains_by_knots",
    "warp_event_by_knots",
    "warp_list_of_events_by_knots",
    "warp_epoch_by_knots",
    "warp_list_of_epochs_by_knots",
    "warp_analogsignal_by_knots",
    "warp_list_of_analogsignals_by_knots",
    "warp_segment_by_events",
]

# TODO documentation
# TODO speed up by parallelization
# TODO put numba into elephant requirements
# TODO take care of numba deprecation warning for untyped lists


@numba.jit(nopython=True)
def warp_sequence_of_time_points(sequence_of_time_points,
                                 original_time_knots,
                                 warping_time_knots):
    """Short summary.

    Inspired from sparsewarp in Alex Williams affinewarp package.

    Parameters
    ----------
    sequence_of_time_points : type
        Description of parameter `sequence_of_time_points`.
    original_time_knots : type
        Description of parameter `original_time_knots`.
    warping_time_knots : type
        Description of parameter `warping_time_knots`.

    Returns
    -------
    type
        Description of returned object.

    """

    n_knots = len(original_time_knots)

    # loop over all spikes
    warped_sequence_of_time_points = []
    for time in sequence_of_time_points:
        for knot_idx in range(n_knots):
            if time <= original_time_knots[knot_idx]:
                break

        if knot_idx == 0:
            x_diff = (original_time_knots[1] - original_time_knots[0])
            y_diff = (warping_time_knots[1] - warping_time_knots[0])
            # cover division by zero
            if x_diff == 0:
                # TODO dig into this!
                slope = 1
            else:
                slope = y_diff / x_diff
        else:
            x_diff = (original_time_knots[knot_idx] -
                      original_time_knots[knot_idx-1])
            y_diff = (warping_time_knots[knot_idx] -
                      warping_time_knots[knot_idx-1])
        # cover division by zero
        if x_diff == 0:
            # TODO dig into this!
            slope = 1
        else:
            slope = y_diff / x_diff

        warped_sequence_of_time_points.append(
            warping_time_knots[knot_idx] +
            slope * (time - original_time_knots[knot_idx]))

    return np.array(warped_sequence_of_time_points).flatten()

# TODO update documentation


def warp_spiketrain_by_knots(spiketrain,
                             original_time_knots,
                             warping_time_knots):
    """Warps a single spiketrains by specifying warping knots.

    This function allows for linear and piecewise linear warping
    of a single spiketrain.
    
    Parameters
    ----------
    spiketrain : type
        Description of parameter `spiketrain`.
    original_time_knots : type
        Description of parameter `original_time_knots`.
    warping_time_knots : type
        Description of parameter `warping_time_knots`.

    Returns
    -------
    warped_spiketrain
        Warped neo.Spiketrain

    """
    warped_spike_times = warp_sequence_of_time_points(spiketrain,
                                                      original_time_knots,
                                                      warping_time_knots)

    warped_spiketrain = neo.SpikeTrain(
        name=f'{spiketrain.name}',
        times=warped_spike_times,
        t_start=warping_time_knots[0],
        t_stop=warping_time_knots[-1],
        units=spiketrain.units)

    warped_spiketrain.annotate(**copy.deepcopy(spiketrain.annotations))
    warped_spiketrain.array_annotate(
        **copy.deepcopy(spiketrain.array_annotations))
    if 'nix_name' in warped_spiketrain.annotations:
        warped_spiketrain.annotations.pop('nix_name')

    return warped_spiketrain

# TODO documentation
# TODO merge list of spiketrains and spiketrain warp?


def warp_list_of_spiketrains_by_knots(list_of_spiketrains,
                                      original_time_knots,
                                      warping_time_knots):

    list_of_warped_spiketrains = []
    for st in list_of_spiketrains:
        warped_st = warp_spiketrain_by_knots(st,
                                             original_time_knots,
                                             warping_time_knots)
        list_of_warped_spiketrains.append(warped_st)
    return list_of_warped_spiketrains

# TODO documentation


def warp_event_by_knots(event,
                        original_time_knots,
                        warping_time_knots):
    warped_event_times = warp_sequence_of_time_points(event,
                                                      original_time_knots,
                                                      warping_time_knots)

    warped_event = neo.Event(
        name=f'{event.name}',
        times=warped_event_times,
        labels=event.labels,
        units=event.units)

    warped_event.annotate(**copy.deepcopy(event.annotations))
    warped_event.array_annotate(**copy.deepcopy(event.array_annotations))
    if 'nix_name' in warped_event.annotations:
        warped_event.annotations.pop('nix_name')

    return warped_event

# TODO documentation
# TODO merge list of spiketrains and spiketrain warp


def warp_list_of_events_by_knots(list_of_events,
                                 original_time_knots,
                                 warping_time_knots):

    list_of_warped_events = []
    for ev in list_of_events:
        warped_ev = warp_event_by_knots(ev,
                                        original_time_knots,
                                        warping_time_knots)
        list_of_warped_events.append(warped_ev)
    return list_of_warped_events

# TODO documentation


def warp_epoch_by_knots(epoch,
                        original_time_knots,
                        warping_time_knots):
    warped_epoch_start_times = warp_sequence_of_time_points(
        epoch,
        original_time_knots,
        warping_time_knots)

    # the warped duration is obtained by warping the original epoch stop
    epoch_stop_time = epoch.times + epoch.durations
    warped_epoch_stop_times = warp_sequence_of_time_points(epoch_stop_time,
                                                           original_time_knots,
                                                           warping_time_knots)

    warped_epoch_durations = warped_epoch_stop_times - warped_epoch_start_times

    warped_epoch = neo.Epoch(
        name=f'{epoch.name}',
        times=warped_epoch_start_times,
        durations=warped_epoch_durations,
        labels=epoch.labels,
        units=epoch.units)

    warped_epoch.annotate(**copy.deepcopy(epoch.annotations))
    warped_epoch.array_annotate(**copy.deepcopy(epoch.array_annotations))
    if 'nix_name' in warped_epoch.annotations:
        warped_epoch.annotations.pop('nix_name')
    return warped_epoch

# TODO documentation
# TODO merge list of spiketrains and spiketrain warp


def warp_list_of_epochs_by_knots(list_of_epochs,
                                 original_time_knots,
                                 warping_time_knots):

    list_of_warped_epochs = []
    for ep in list_of_epochs:
        warped_ep = warp_epoch_by_knots(ep,
                                        original_time_knots,
                                        warping_time_knots)
        list_of_warped_epochs.append(warped_ep)
    return list_of_warped_epochs


def warp_analogsignal_by_knots(analogsignal,
                               original_time_knots,
                               warping_time_knots,
                               irregular_signal=False,
                               sampling_period=None,
                               interpolation_kind='linear'):
    # TODO clean up documentation
    """warp analogsignal according to warping knots
    
    The time stamps of the analogsignal will be warped, while keeping the 
    signal untouched. By default, the signal will then be resampled to
    a regular sampling period by interpolating the signal onto newly
    defined time stamps.

    Parameters
    ----------
    analogsignal : neo.Analogsignal
        Signal that should be warped.
    original_time_knots : list
    warping_time_knots : list
    irregular_signal : bool, optional
        If True, only the time points are warped and together with the signal
        stored in a neo.IrregularlySampledSignal, by default False
    sampling_period : [pq.Quantity], optional
        Default takes the same sampling period as the original 
        analogsignal, by default None
    interpolation_kind : str, optional
        by default 'linear'

    Returns
    -------
    neo.Analogsignal or neo.IrregularlySampledSignal
        warped_analogsignal
    """    

    warped_times = warp_sequence_of_time_points(analogsignal.times,
                                                original_time_knots,
                                                warping_time_knots)
    
    warped_irregularlysampledsignal = neo.IrregularlySampledSignal(
        times=warped_times,
        signal=analogsignal,
        units=analogsignal.units,
        time_units=analogsignal.times.units)
    
    if irregular_signal:
        warped_analogsignal = warped_irregularlysampledsignal
    else:
        interpolate_function = interp1d(
            warped_irregularlysampledsignal.times,
            warped_irregularlysampledsignal,
            kind=interpolation_kind,
            axis=0) # interpolate along time axis
        
        if sampling_period is None:
            sampling_period = analogsignal.sampling_period
        
        new_time_points = np.arange(
            warped_irregularlysampledsignal.t_start.rescale(pq.s),
            warped_irregularlysampledsignal.t_stop.rescale(pq.s),
            sampling_period)

        warped_interpolated_data = interpolate_function(new_time_points)
        warped_analogsignal = neo.AnalogSignal(
            signal=warped_interpolated_data,
            units=analogsignal.units,
            time_units=analogsignal.times.units,
            t_start=warped_irregularlysampledsignal.t_start.rescale(pq.s),
            sampling_period=sampling_period)
        
    warped_analogsignal.name = f'{analogsignal.name}'
    warped_analogsignal.annotate(**copy.deepcopy(analogsignal.annotations))
    warped_analogsignal.array_annotate(
        **copy.deepcopy(analogsignal.array_annotations))
    if 'nix_name' in warped_analogsignal.annotations:
        warped_analogsignal.annotations.pop('nix_name')
    return warped_analogsignal

# TODO merge list of spiketrains and spiketrain warp


def warp_list_of_analogsignals_by_knots(list_of_analogsignals,
                                        original_time_knots,
                                        warping_time_knots,
                                        irregular_signal=False,
                                        sampling_period=None,
                                        interpolation_kind='linear'):

    list_of_warped_analogsignals = []
    for anasig in list_of_analogsignals:
        warped_anasig = warp_analogsignal_by_knots(anasig,
                                                   original_time_knots,
                                                   warping_time_knots,
                                                   irregular_signal,
                                                   sampling_period,
                                                   interpolation_kind)
        list_of_warped_analogsignals.append(warped_anasig)
    return list_of_warped_analogsignals


def get_warping_knots(segment,
                      event_name,
                      new_events_dictionary,
                      return_labels_of_warped_events=False):
    
    # get original event times
    events = utils.get_events(
        container=segment,
        # name=event_name,
        labels=list(new_events_dictionary.keys())
        )
    
    # merge returned events in case the requested events come from
    # different neo.Events
    if len(events) > 1:
        for i in range(1, len(events)):
            events[0] = events[0].merge(events[i])

    sort_indices = np.argsort(events[0].times)
    original_event_labels = events[0].labels[sort_indices]
    original_event_times = events[0].times[sort_indices]

    labels_of_warped_events = list(new_events_dictionary.keys())
    new_event_times = [time.rescale(pq.s).magnitude.item() for time 
                       in new_events_dictionary.values()] * pq.s
    if return_labels_of_warped_events:
        return original_event_times, new_event_times, labels_of_warped_events
    return original_event_times, new_event_times


def cut_segment_to_warping_time_range(segment,
                                      event_name,
                                      new_events_dictionary):

    starting_warping_knot = utils.get_events(
        container=segment,
        name=event_name,
        labels=list(new_events_dictionary.keys())[0])[0]

    end_warping_knot = utils.get_events(
        container=segment,
        name=event_name,
        labels=list(new_events_dictionary.keys())[-1])[0]

    warping_epoch = utils.add_epoch(
        segment,
        event1=starting_warping_knot,
        event2=end_warping_knot,
        attach_result=False,
        name='Warping Epoch')

    # TODO fails if analogsignal t_start is later than first event
    # or t_stop earlier than last event
    # print(warping_epoch.times[0], warping_epoch.durations[0], segment.t_start, segment.t_stop, segment.annotations['trial_number'])
    
    warping_segment = utils.cut_segment_by_epoch(
        seg=segment,
        epoch=warping_epoch,
        reset_time=True)[0]

    return warping_segment

# TODO write another function for just warping t_stop!
def warp_segment_by_events(
        segment,
        event_name,
        new_events_dictionary,
        irregular_signal=False,
        sampling_period=None,
        interpolation_kind='linear'):
    
    """Warp a neo.Segment by specifying (warped) times of events.

    Parameters
    ----------
    segment : neo.Segment
        The neo.Segment that shall be warped.
    event_name : str
        Name of the neo.Event that `segment` contains and that the events
        for the `new_events_dictionary` are taken from.
    new_events_dictionary : dict
        The `new_events_dictionary` contains as keys the labels of
        the events that shall be warped. Their corresponding values
        are the new (pre-defined) event times.
        Internally the original event times will be obtained from the
        segment.
    irregular_signal : bool, optional
        If True, only the time points are warped and together with the signal
        stored in a neo.IrregularlySampledSignal, by default False
    sampling_period : [pq.Quantity], optional
        Default takes the same sampling period as the original 
        analogsignal, by default None
    interpolation_kind : str, optional
        by default 'linear'

    Returns
    -------
    original_event_times : list
        List containing the original event times.
    new_event_times : list
        List containing the new event times.
    warped_segment : neo.Segment
        Warped neo.Segment.
    """
    
    segment = cut_segment_to_warping_time_range(segment,
                                                event_name,
                                                new_events_dictionary)

    (original_event_times,
     new_event_times,
     new_event_labels) = get_warping_knots(segment,
                                           event_name,
                                           new_events_dictionary,
                                           return_labels_of_warped_events=True)    

    assert(len(original_event_times) == len(new_event_times))

    # create a new neo.Segment
    # TODO proper naming only works for a list of segments if each
    # segment has a unique name
    warped_segment = neo.Segment(
        name=f'{segment.name}'
    )
    warped_segment.annotate(**copy.deepcopy(segment.annotations))
    warped_segment.annotate(original_event_times=original_event_times,
                            new_event_times=new_event_times,
                            warped_event_labels=new_event_labels)
    if 'nix_name' in warped_segment.annotations:
        warped_segment.annotations.pop('nix_name')

    warped_spiketrains = warp_list_of_spiketrains_by_knots(
        segment.spiketrains,
        original_event_times,
        new_event_times)
    warped_segment.spiketrains = warped_spiketrains

    warped_events = warp_list_of_events_by_knots(segment.events,
                                                 original_event_times,
                                                 new_event_times)
    warped_segment.events = warped_events

    warped_epochs = warp_list_of_epochs_by_knots(segment.epochs,
                                                 original_event_times,
                                                 new_event_times)
    warped_segment.epochs = warped_epochs

    warped_analogsignals = warp_list_of_analogsignals_by_knots(
        segment.analogsignals,
        original_event_times,
        new_event_times,
        irregular_signal,
        sampling_period,
        interpolation_kind)
    if irregular_signal:
        warped_segment.irregularlysampledsignals = warped_analogsignals
    else:
        warped_segment.analogsignals = warped_analogsignals

    warped_segment.create_relationship()

    return original_event_times, new_event_times, warped_segment


def warp_list_of_segments_by_events(
        list_of_segments,
        event_name,
        new_events_dictionary,
        irregular_signal=False,
        sampling_period=None,
        interpolation_kind='linear'):

    new_block = neo.Block(name='Block warped by events')
    for seg_idx, segment in enumerate(list_of_segments):
        print(f'Currently warping segment {seg_idx}', end="\r")
        knots_x, knots_y, warped_segment = warp_segment_by_events(
            segment,
            event_name,
            new_events_dictionary,
            irregular_signal,
            sampling_period,
            interpolation_kind)
        new_block.segments.append(warped_segment)

    new_block.create_relationship()
    return new_block
