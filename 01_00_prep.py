#!/usr/bin/env python
# coding: utf-8

# # Filter, resample, and organize the channels

# In[ ]:


import mne
from os import listdir
from os.path import isdir, join

import re
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


mne.utils.set_config('MNE_USE_CUDA', 'True')


# Expecting data folder directory structure as described in 00_00_convert

# In[ ]:


#root_dir = str(Path.home())
root_dir = str("/media/Linux6_Data/DATA/SFB2")
proc_dir = join(root_dir, "proc") # working directory

sub_dirs = [item for item in listdir(proc_dir) if isdir(join(proc_dir, item))]

l_freq = 0.1
h_freq = 100
n_jobs = 24 #= "cuda" # change this to 1 or some higher integer if you don't have CUDA
sfreq = 200 # frequency to resample to

include = [["1002", "T2"]]
overwrite = True

testing = False # for testing, only run once
number_of_preprocessed_files = 0


# In[ ]:


for subdir in sub_dirs:

    print(f"{subdir}".center(80, '-'))
    proclist = listdir(join(proc_dir, subdir)) # and in proc directory

    for file in proclist:

        # if we are testing, only apply preprocessing to one file
        if testing and number_of_preprocessed_files > 0:
            continue

        # find out subject id (4 digits) and condition (T1, T2, T3, T4) from file name
        match = re.search("NAP_(?P<subj>\\d{4})_(?P<cond>T\\d{1})-raw.fif", file)
        if not match:
            continue
        subj = match.group('subj')
        cond = match.group('cond')
        if include and [subj, cond] not in include:
            continue
        
        print(f"Processing {subj} {cond}")

        # set the file name of the output .fif file and check if it already exists in proc_dir
        #outfile =  f"NAP_{subj}_{cond}-01_00_prep.fif"
        outfile =  f"NAP_{subj}_{cond}-rp.fif"
        if outfile in proclist and not overwrite:
            print(f"{outfile} already exists in processing dir. Skipping...")
            continue

        number_of_preprocessed_files += 1

        # actually do the preprocessing
        try:
            raw = mne.io.Raw(join(proc_dir, subdir, file), preload=True)

            # TODO: Low/Highpass filter (possibly apply a highpass filter 0.1..)
            #raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)

            # apply a notch filter to remove power line artifact
            raw.notch_filter(np.arange(50, h_freq+50, 50), n_jobs=n_jobs)
    
            # resample to lower frequency 
            raw.resample(sfreq, n_jobs=n_jobs)

            # create EOG/EMG channels
            if "HEOG" not in raw.ch_names:
                raw = mne.set_bipolar_reference(raw, "Li", "Re", ch_name="HEOG")
                raw.set_channel_types({"HEOG":"eog"})
            if "Mov" not in raw.ch_names:
                raw = mne.set_bipolar_reference(raw, "MovLi", "MovRe", ch_name="Mov")
                raw.set_channel_types({"Mov":"emg"})
            if "VEOG" not in raw.ch_names and "Vo" in raw.ch_names and "Vu" in raw.ch_names:
                raw = mne.set_bipolar_reference(raw, "Vo", "Vu", ch_name="VEOG")
                raw.set_channel_types({"VEOG":"eog"})

            raw.save(join(proc_dir, subdir, outfile), overwrite=overwrite)

        except Error as e:
            print(f"!!!   Having a problem with {subj} {cond}")
            print(e)
    

