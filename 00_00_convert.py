#!/usr/bin/env python
# coding: utf-8

# # Convert
# 
# From Brainvision format to .fif used by MNE Python.

# In[ ]:


import mne
from os import listdir
from os.path import isdir, join
import re
from pathlib import Path


# As a root dir set home (there best add a symlink to relevant data)
# 
# expected data folder directory structure is as follows
# 
# * data/
#     * raw/
#         * sham 
#         * sotDCS_anod 
#         * sotDCS_cat 
#         * tDCS
#     * proc/
#         * sham 
#         * sotDCS_anod 
#         * sotDCS_cat 
#         * tDCS

# In[ ]:


root_dir = str("/media/Linux6_Data/DATA/SFB2")

raw_dir = join(root_dir, "raw") # get brainvision raw files from here
proc_dir = join(root_dir, "proc") # save the processed .fif files here

sub_dirs = [item for item in listdir(raw_dir) if isdir(join(raw_dir, item))]

include = [["1002", "T2"]]

overwrite = True


# Files in brain vision format (.eeg, .vhdr, .vmrk)
# 
# individual files are named analogous to "NAP_0000_T0.eeg"

# In[ ]:


for subdir in sub_dirs:

    print(f"{subdir}".center(80, '-'))
    filelist = listdir(join(raw_dir, subdir)) # get list of all files in raw directory
    proclist = listdir(join(proc_dir, subdir)) # and in proc directory

    for file in filelist:

        # find out subject id (4 digits) and condition (T1, T2, T3, T4) from file name
        match = re.search("NAP_(?P<subj>\\d{4})_(?P<cond>T\\d{1}).vhdr", file)
        if not match:
            continue
        subj = match.group('subj')
        cond = match.group('cond')
        if include and [subj, cond] not in include:
            continue

        print(f"Processing {subj} {cond}")

        # set the file name of the raw.fif file and check if it already exists in proc_dir
        outfile =  f"NAP_{subj}_{cond}-raw.fif"
        if outfile in proclist and not overwrite:
            print(f"{outfile} already exists in processing dir. Skipping...")
            continue

        # actually do the conversion
        try:
            raw = mne.io.read_raw_brainvision(join(raw_dir, subdir, file),preload=True) 
            raw.save(join(proc_dir, subdir, outfile), overwrite=overwrite)
        except Exception as e:
            print (e)
            print(f"!!!   Having a problem with {subj} {cond}")
    


# 
