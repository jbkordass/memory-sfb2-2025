#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
from os import listdir
import re
from os.path import isdir, join
import pandas as pd


# 
# <br>
# concatenate all epochs into one file<br>
# 

# In[ ]:


root_dir = "/media/Linux6_Data/DATA/SFB2"
proc_dir = join(root_dir, "proc")

sub_dirs = [item for item in listdir(proc_dir) if isdir(join(proc_dir, item))]


# In[ ]:


overwrite = True
epos = []
for subdir in sub_dirs:

    print(f"{subdir}".center(80, '-'))
    proclist = listdir(join(proc_dir, subdir)) # and in proc directory
    save_dir = join(proc_dir,subdir)

    
    for filename in proclist:
        this_match = re.match("osc_NAP_(\d{4})_(.*)_(.*)_(.*)_(.*)-epo.fif", filename)
        if not this_match:
            continue
        #(subj, cond, chan, osc_type) = this_match.groups()
        epo = mne.read_epochs(join(proc_dir, subdir, filename))
        epos.append(epo)

grand_epo = mne.concatenate_epochs(epos)
grand_epo.save(join(proc_dir, "grand-epo.fif"), overwrite=True)

