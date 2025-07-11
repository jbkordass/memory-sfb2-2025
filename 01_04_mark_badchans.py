#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
from os import listdir
import re
from anoar import BadChannelFind
from os.path import isdir, join
from pathlib import Path
import numpy as np


# In[ ]:


root_dir = str("/media/Linux6_Data/DATA/SFB2")
proc_dir = join(root_dir, "proc") # working directory

ignordir = ["SO_epochs_NOdetr","SO_epochs_detr","figs"]

sub_dirs = [item for item in listdir(proc_dir) if isdir(join(proc_dir, item)) and item not in ignordir]

specCase = [
            #["1019", "T1"],
           # ["1023", "T4"],
           # ["1059", "T1"]
]
include = [
            ["1002", "T2"],
            #["1023", "T4"],
            #["1059", "T1"]
]
overwrite = True

n_jobs = 24

testing = True # for testing, only run once
number_of_preprocessed_files = 0


# In[ ]:


number_of_preprocessed_files = 0

for subdir in sub_dirs:

    print(f"{subdir}".center(80, '-'))
    proclist = listdir(join(proc_dir, subdir)) # and in proc directory

    for file in proclist:
        this_match = re.match("NAP_(?P<subj>\\d{4})_(?P<cond>T\\d{1})-rpac.fif", file)
        if this_match:
            subj, cond = this_match.group(1), this_match.group(2)

            if include and [subj, cond] not in include:
                continue

            outname = f"NAP_{subj}_{cond}-rpacb.fif"
            if outname in proclist and not overwrite:
                print(f"{outname} already exists. Skipping...")
                continue

            raw = mne.io.Raw(join(proc_dir, subdir, file),preload=True)
            #raw_clean = raw.copy()
            # Extract the annotations you want to remove
            bad_annotations = [annot for annot in raw.annotations if 'BAD' in annot['description'] or 'ToEnd' in annot['description']]
            # Onsets and durations of the BAD time spans
            omit_ranges = [(annot['onset'], annot['duration']) for annot in bad_annotations]
            # Get the data as a Numpy array
            data = raw.get_data()

            # Create a mask to keep the time points
            time_points = raw.times
            mask = np.ones(data.shape[1], dtype=bool)  

            # Set the mask for the time spans to be removed to False
            for onset, duration in omit_ranges:
                start_idx = np.searchsorted(time_points, onset)
                end_idx = np.searchsorted(time_points, onset + duration)
                mask[start_idx:end_idx] = False  # Ignore these time points

            # Filter the raw data based on the mask
            data_cleaned = data[:, mask]

            # Create a new Raw object with the cleaned data
            info = raw.info
            raw_cleaned = mne.io.RawArray(data_cleaned, info)
            if specCase:
                raw_cleaned.filter(l_freq=None, h_freq=30, n_jobs=n_jobs)
                picks = mne.pick_types(raw_cleaned.info, eeg=True)
                bcf = BadChannelFind(picks, thresh=0.2)
                bad_chans = bcf.recommend(raw_cleaned)
                print('Bads in :' + f"NAP_{subj}_{cond}")
                print(bad_chans)
                raw.info["bads"].extend(bad_chans)
                raw.save(join(proc_dir, subdir,outname),overwrite=overwrite)
                continue

            picks = mne.pick_types(raw_cleaned.info, eeg=True)
            bcf = BadChannelFind(picks, thresh=0.5)
            bad_chans = bcf.recommend(raw_cleaned)
            print('Bads in :' + f"NAP_{subj}_{cond}")
            print(bad_chans)
            raw.info["bads"].extend(bad_chans)
            raw.save(join(proc_dir, subdir,outname),overwrite=overwrite)

