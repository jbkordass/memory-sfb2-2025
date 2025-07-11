#!/usr/bin/env python
# coding: utf-8

# ## Artefact detection
# Roughly following the ideas about artifact detection outlined by Leo in prep_test.py

# In[ ]:


import mne
from os import listdir
from os.path import isdir, join

import re
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# In[ ]:


root_dir = "/media/Linux6_Data/DATA/SFB2"
proc_dir = join(root_dir, "proc") # working directory

sub_dirs = [item for item in listdir(proc_dir) if isdir(join(proc_dir, item))]

overwrite = True

testing = True # for testing, only run once


# In[ ]:


def annot_grad(raw, thresh=None, extend = 0.2, channel= None, start= 0, stop = None):
    """
    Find and annotate gradient artifacts within the indices range (start, stop)
    """
    data = raw.get_data(return_times=True, picks = channel, start= start, stop=stop)
    times = data[1]
    if channel:
        channel_list = channel
    else:
        channel_list = raw.ch_names
    annot_dict = dict(onset= list(), duration=list(), description = list(), orig_time = list(), ch_names = list())

    pend = np.zeros((len(channel_list),1))
    
    derive = np.diff(data[0], axis=1, prepend=pend, append= pend)[:,0:-1]
    if not derive.any():
        return None

    # if no threshold specified, try to guess one
    if thresh==None: 
        median = np.median(derive, axis=1)
        firstq, thirdq = np.percentile(derive, [25, 75], axis = 1)
        interquar = thirdq - firstq
        thresh = median + 20*interquar

    
    mask = np.zeros(np.shape(derive))
    mask = np.where(abs(derive)>thresh[:, None], 1, mask)
    segm = np.diff(mask, axis=1, prepend = pend, append = pend)[:,0:-1]

    onsets = np.argwhere(segm == 1)
    offsets = np.argwhere(segm == -1)
    if len(onsets)!=len(offsets):
        print(np.shape(segm))
    print(len(onsets),len(offsets))

    new_on = [[] for _ in range(len(channel_list))]
    new_off = [[] for _ in range(len(channel_list))]
    
    for i in range(len(onsets)):
        new_on[onsets[i][0]].append(onsets[i][1])
    for i in range(len(offsets)):
        new_off[offsets[i][0]].append(offsets[i][1])

    for i, chan in enumerate(channel_list):
        if len(new_on[i]) > len(new_off[i]):
            new_off[i].append(new_on[i][-1])
        annot_dict['onset']+=list(times[new_on[i]]-extend)
        annot_dict['duration']+=list(times[new_off[i]]-times[new_on[i]]+2*extend)
        annot_dict['description']+=list(np.repeat('BAD_step_'+chan, len(new_on[i])))
        annot_dict['orig_time']+=list(np.repeat(raw.info['meas_date'], len(new_on[i])))
        annot_dict['ch_names']+=list(np.tile(channel_list[i], (len(new_on[i]),1)))
    for key, value in annot_dict.items():
        annot_dict[key] = np.asarray(value)
    annot = mne.Annotations(onset= annot_dict['onset'], duration= annot_dict['duration'], description= annot_dict['description'], orig_time = raw.info['meas_date'])

    return annot



def annot_abs(raw, thresh= 7.5e-4, extend = 0.2, channel= None, start = 0, stop = None):
    """
    Find and annotate amplitude artifacts within the indices range (start, stop)
    """

    data = raw.get_data(return_times=True, picks = channel, start = start, stop= stop)
    times = data[1]
    if channel:
        channel_list = channel
    else:
        channel_list = raw.ch_names
    bad_ints = dict(onset= list(), duration=list(), description = list(), orig_time = list(), ch_names = list())
    

    pend = np.zeros((len(channel_list),1))
    mask = np.zeros(np.shape(data[0]))
    mask = np.where(abs(data[0])>thresh, 1, mask)
    segm = np.diff(mask, axis=1, prepend= pend, append=pend)[:,0:-1]
    onsets = np.argwhere(segm == 1)
    offsets = np.argwhere(segm == -1)
    new_on = [[] for _ in range(len(channel_list))]
    new_off = [[] for _ in range(len(channel_list))]
    for i in range(len(onsets)):
        new_on[onsets[i][0]].append(onsets[i][1])
    for i in range(len(offsets)):
        new_off[offsets[i][0]].append(offsets[i][1])
    for i, chan in enumerate(channel_list):
        if len(new_on[i]) > len(new_off[i]):
            new_off[i].append(new_on[i][-1])
        bad_ints['onset']+=list(times[new_on[i]]-extend)
        bad_ints['duration']+=list(times[new_off[i]]-times[new_on[i]]+2*extend)
        bad_ints['description']+=list(np.repeat('BAD_amplitude_'+chan, len(new_on[i])))
        bad_ints['orig_time']+= list(np.repeat(raw.info['meas_date'], len(new_on[i])))
        bad_ints['ch_names']+=list(np.tile(channel_list[i], (len(new_on[i]),1)))
    for key, value in bad_ints.items():
        bad_ints[key] = np.asarray(value)
    annot= mne.Annotations(onset= bad_ints['onset'], duration= bad_ints['duration'], description= bad_ints['description'], ch_names = bad_ints['ch_names'], orig_time = raw.info['meas_date'])
    return annot


# In[ ]:


def intervals_from_annotations(raw, description, end_interval = True):
    """
    Returns intervals (start, stop) indices of annotations based on an expression of annotation descriptions

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data
    description : list of str
        The descriptions of the annotations to look for (use regex notation, e.g. ending with ".*" to avoid specifying each individually)
    end_interval : bool
        if true, we add an interval from the last annotation index matching the description to the end of the data

    Returns
    -------
    list of tuples (start int, end int, description str)
    """
    times = raw.get_data(return_times = True)[1]
    annotations = raw.annotations.copy()

    annotation_intervals = []
    # cycle through all annotations and find the ones that match elements in the description list
    for annot in annotations:
        for j in range(len(description)):
            if re.match(description[j], annot['description']):
                indices = raw.time_as_index([annot['onset'], annot['onset'] + annot['duration']])
                annotation_intervals.append((indices[0], indices[1], annot['description']))
    
    if end_interval:
        annotation_intervals.append((annotation_intervals[-1][1]+1, len(times)-1, 'End'))
    return annotation_intervals


# In[ ]:


number_of_preprocessed_files = 0

for subdir in sub_dirs:

    print(f"{subdir}".center(80, '-'))
    proclist = listdir(join(proc_dir, subdir)) # and in proc directory

    for file in proclist:

        # if we are testing, only apply step to one file
        if testing and number_of_preprocessed_files > 0:
            continue

        # find out subject id (4 digits) and condition (T1, T2, T3, T4) from file name
        match = re.search("NAP_(?P<subj>\\d{4})_(?P<cond>T\\d{1})-rpacd.fif", file)
        if not match:
            continue
        subj = match.group('subj')
        cond = match.group('cond')
        print(f"Processing {subj} {cond}")

        # set the file name of the output .fif file and check if it already exists in proc_dir
        outfile =  f"NAP_{subj}_{cond}-rpacda.fif"
        if outfile in proclist and not overwrite:
            print(f"{outfile} already exists in processing dir. Skipping...")
            continue

        number_of_preprocessed_files += 1

        # now do the actual processing step
        # ---------------------------------

        try:
            raw = mne.io.Raw(join(proc_dir, subdir, file), preload=True)
        except:
            print(f"Error loading raw for {subj} {cond}")
            continue


        # get data intervals by the newly added annotations
        intervals = intervals_from_annotations(raw, ['Pre_Stimulation','Post_Stimulation.*'], end_interval=True)

        # apply artifact annotation
        for start, stop, desc in intervals:

            old_annotations= raw.annotations.copy()
            new_annotations= []
            new_annotations.append(annot_grad(raw, start = start, stop = stop))
            new_annotations.append(annot_abs(raw, start = start, stop = stop))
            new_annotations.append(annot_abs(raw, thresh=15e-5, channel = ['Mov'], start = start, stop = stop))

            for annot in new_annotations:
                if annot:
                    old_annotations = old_annotations.__add__(annot)
                    print(annot)
            raw.set_annotations(old_annotations)

        raw.save(join(proc_dir, subdir, outfile), overwrite=overwrite)


# In[ ]:




