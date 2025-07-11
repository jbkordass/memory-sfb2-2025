#!/usr/bin/env python
# coding: utf-8

# ## Robust detrending and artefact detection
# ~~Roughly following the ideas outlined by Leo in prep_test.py~~
# 
# Simply apply robust detrending everywhere.. that seems a whole lot cleaner.
# 
# Leos solution also did not really work well with the cutout version of the intervals. (Somehow times in the annotations did not map to time points after cropping.)

# Use masked robust detrending according to https://doi.org/10.1016/j.jneumeth.2021.109080
# 
# - Masked events are pre/post stimulation annotations (not the last one that goes unitl the end of the data)
# - Detrend is only calculated over masked events but applied to the whole data
# - `create_masked_weight` creates a weight that is passed to the `detrend` function
# - cf. https://nbara.github.io/python-meegkit/modules/meegkit.detrend.html

# In[ ]:


import mne
from os import listdir
from os.path import isdir, join

import re
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from meegkit.detrend import detrend, create_masked_weight


# In[ ]:


root_dir = "/media/Linux6_Data/DATA/SFB2"
proc_dir = join(root_dir, "proc") # working directory

sub_dirs = [item for item in listdir(proc_dir) if isdir(join(proc_dir, item))]

# parameters
analy_duration = 60 # duration of the (pre/poststim) analysis window

# detrending parameters
detrend_orders = [1, 6]

overwrite = True

testing = False # for testing, only run once

include = []
exclude = []


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
        match = re.search("NAP_(?P<subj>\\d{4})_(?P<cond>T\\d{1})-rpacb.fif", file)
        if not match:
            continue
        subj = match.group('subj')
        cond = match.group('cond')
        print(f"Processing {subj} {cond}")

        if [subj, cond] in exclude:
            continue
        if include and [subj, cond] not in include:
            continue

        # set the file name of the output .fif file and check if it already exists in proc_dir
        outfile =  f"NAP_{subj}_{cond}-rpacbd.fif"
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

        # masked robust detrending
        masked_events = np.array([])
        annotations = raw.annotations
        for idx, annot in enumerate(annotations):
            if re.match("Pre_Stimulation", annot['description']):
                masked_events = np.append(masked_events, annot['onset'])
            if re.match("Post_Stimulation", annot['description']):
                masked_events = np.append(masked_events, annot['onset'])
            if re.match("Post_Stimulation_ToEnd", annot['description']):
                duration = annot['duration']
                for i in range(1, int(duration/analy_duration)):
                    masked_events = np.append(masked_events, annot['onset'] + i*analy_duration)

        X = raw.get_data().T # transpose so the data is organized time-by-channels

        weight = create_masked_weight(X, masked_events, tmin = 0, tmax = analy_duration, sfreq = raw.info['sfreq'])

        for order in detrend_orders:
            X, _, _ = detrend(X, order=order, w=weight) 

        raw._data = X.T  # overwrite raw data

        raw.save(join(proc_dir, subdir, outfile), overwrite=overwrite)

