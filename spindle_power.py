#!/usr/bin/env python
# coding: utf-8

# ## Extract spindle power in poststim intervals and during positive half wave of SOs

# In[ ]:


import mne
from os import listdir
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
#from anoar import BadChannelFind
from scipy.signal import find_peaks
from os.path import isdir, join
plt.ion()
from scipy.signal import welch


# In[ ]:


with_so = False # decide whether to find spindles during SOs or in 1 min post-Stim

if with_so:
    root_dir = "/media/Linux6_Data/johnsm"
    proc_dir = join(root_dir, "outputs") # working directory
else: 
    root_dir = "/media/Linux6_Data/DATA/SFB2"
    proc_dir = join(root_dir, "proc") # working directory

sub_dirs = [item for item in listdir(proc_dir) if isdir(join(proc_dir, item))]

overwrite = True

testing = False # for testing, only run once

include = []
exclude = []

chan_groups_A = {"frontal":["Fz"],
               "parietal":["Cz"]}

chan_groups_B = {"frontal":["Fz"],     # ROI alternative B, if chan_group_A is in 'bads'
               "parietal":["CP1"]}

chan_groups_C = {"frontal":["Fz"],     # ROI alternative C, if chan_group and chan_groups_B is in 'bads' 
               "parietal":["CP2"]}

chan_groups_D = {"frontal":["FC1"],
               "parietal":["Cz"]}

chan_groups_E = {"frontal":["FC1"],     # ROI alternative B, if chan_group_A is in 'bads'
               "parietal":["CP1"]}

chan_groups_F = {"frontal":["FC1"],     # ROI alternative C, if chan_group and chan_groups_B is in 'bads' 
               "parietal":["CP2"]}

chan_groups_G = {"frontal":["FC2"],
               "parietal":["Cz"]}

chan_groups_H = {"frontal":["FC2"],     # ROI alternative B, if chan_group_A is in 'bads'
               "parietal":["CP1"]}

chan_groups_I = {"frontal":["FC2"],     # ROI alternative C, if chan_group and chan_groups_B is in 'bads' 
               "parietal":["CP2"]}



chan_groups = [chan_groups_A,chan_groups_B,chan_groups_C,chan_groups_D,chan_groups_E, chan_groups_F,chan_groups_G,chan_groups_H,chan_groups_I]

amp_percentile = 75 # lower percentile bound for amplitude of SOs (= |peak| + |trough|)
min_samples = 10 # min duration between zero crossings in an oscillation in terms of samples (1 sample = 1/sfreq s)

minmax_freqs = [(0.16, 1.25), (0.75, 4.25)] # min_max freqs for SO and delta
minmax_times = [(0.8, 2), (0.25, 1)] # min_max times for SO and delta
osc_types = ["SO"] # "SO", "delta" would be an option, here only use SO (above freqs/times values for delta are irrelevant...)

channel_mode = "averaged"


# ## YASA based spindle detection

# In[ ]:


get_ipython().run_line_magic('pip', 'install yasa #install yasa https://github.com/raphaelvallat/yasa')


# Don't forget to restart the kernel after installing!

# In[ ]:


import yasa 

def spindles_yasa(filename,subj,cond, with_so,fp):

    if not with_so:
        #frontal/parietal channel selection (as during SO detection, only needed when inspecting data before that point)
    
        c_groups_alt = chan_groups.copy()
        
        for c_groups in c_groups_alt:
            print("Try ROI: " + str(c_groups))
            raw = mne.io.read_raw(join(proc_dir, subdir, file), preload=True)
            picks = mne.pick_types(raw.info, eeg=True)
            passed = np.zeros(len(c_groups), dtype=bool)
            #print(len(passed))
            for idx, (k,v) in enumerate(c_groups.items()):
                pick_list = [vv for vv in v if vv not in raw.info["bads"]]
                print("Try CH: " + k + " -- " + str(pick_list))
                if not len(pick_list):
                    print("No valid channels")
                    #skipped["chan"].append("{} {} {} {}".format(subj, cond ,k , v))
                    continue
                avg_signal = raw.get_data(pick_list).mean(axis=0, keepdims=True)
                avg_info = mne.create_info([k], raw.info["sfreq"], ch_types="eeg")
                avg_raw = mne.io.RawArray(avg_signal, avg_info)
                raw.add_channels([avg_raw], force_update_info=True)
                passed[idx] = 1
            if all(passed):
                # ROIs only, drop everything else
                raw.pick_channels(list(c_groups.keys()))
                break
            else:
                continue #mandatory
   
        if not all(passed):
            print("Could not produce valid ROIs")
            #skipped["ROI"].append("{} {}".format(subj, cond))
            return
        raw_eeg=raw.copy()
    else:
        raw_eeg=mne.io.read_raw(filename, preload=True)

    # cycle through annotations in raw_eeg and replace "Active" and "Buffer" with "BAD" make sure these do not get picked up by yasa
    new_annots_1 = mne.Annotations(onset=[], duration=[], description=[])
    for annot in raw_eeg.annotations:
        if "Active" in annot["description"]:
            new_annots_1.append(annot["onset"], annot["duration"], annot["description"].replace("Active", "BAD"))
        elif "Buffer" in annot["description"]:
            new_annots_1.append(annot["onset"], annot["duration"], annot["description"].replace("Buffer", "BAD"))
        else:
            new_annots_1.append(annot["onset"], annot["duration"], annot["description"])
    raw_eeg.set_annotations(new_annots_1)

    #collecting data depending on whether we want to find spindles in the post-stim periods or during the Positive HalfWave of the SOs
    if not with_so:
    # collect data from all Post Stim periods
        datafrontal = np.ndarray(shape=(1,))
        dataparietal = np.ndarray(shape=(1,))
        for annot in raw_eeg.annotations:
            if "Post" in annot["description"]:
            #if "Pre" in annot["description"]:
                start = annot["onset"]
                end = start + annot["duration"]
                datafrontal = np.append(datafrontal, raw_eeg.get_data(tmin=start, tmax=end)[0] * 1000000) # convert from V to uV (yasa)
                dataparietal = np.append(dataparietal, raw_eeg.get_data(tmin=start, tmax=end)[1] * 1000000)
    else:
        data = np.ndarray(shape=(1,))
        for annot in raw_eeg.annotations:
            if "PosHalfWave" in annot["description"]:
                start = annot["onset"]
                end = start + annot["duration"]
                data = np.append(data, raw_eeg.get_data(tmin=start, tmax=end) * 1000000) # convert from V to uV (yasa)

    #calculate bandpower using the function implemented in yasa, for both parietal and frontal 
    if fp == 1:
        fp = "frontal"
    if fp == 2:
        fp = "parietal"
    if with_so:
        bp = yasa.bandpower(data,raw_eeg.info["sfreq"], bands = [(0.5,1,"SO"),(12,16,"spindles")])
        if len(bp)>0:
            bp.drop(bp.columns[0],axis = 1)
            bp = bp.assign(area = [fp]*len(bp))
    else:
        bp1 = yasa.bandpower(datafrontal,raw_eeg.info["sfreq"], bands = [(0.5,1,"SO"),(12,16,"spindles")])
        if len(bp1)>0:
            bp1.drop(bp1.columns[0],axis = 1)
            bp1 = bp1.assign(area = ["frontal"])
        bp2 = yasa.bandpower(dataparietal,raw_eeg.info["sfreq"], bands = [(0.5,1,"SO"),(12,16,"spindles")])
        if len(bp2)>0:
            bp2.drop(bp2.columns[0],axis = 1)
            bp2 = bp2.assign(area = ["parietal"])
        bp = pd.concat([bp1, bp2],axis=0)
        

    return bp



# In[ ]:


number_of_preprocessed_files = 0
if with_so:
    file_name = "so"
else:
    file_name = "poststim"
    #file_name = "prestim"

all_bp = pd.DataFrame()

for subdir in sub_dirs:
    if subdir == "tDCS":
        continue

    print(f"{subdir}".center(80, '-'))
    proclist = listdir(join(proc_dir, subdir)) # and in proc directory

    for file in proclist:

        # if we are testing, only apply step to one file
        if testing and number_of_preprocessed_files > 0:
            continue    #break??

        # find out subject id (4 digits) and condition (T1, T2, T3, T4) from file name
        if with_so:
            match = re.search("osc_NAP_(?P<subj>\\d{4})_(.*)-raw.fif", file)
            if not match:
                continue  
        else:
            #match = re.search("NAP_(?P<subj>\\d{4})_(.*)-rpacbi.fif", file)
            match = re.search("NAP_(?P<subj>\\d{4})_(.*)-rpac.fif", file)
            if not match:
                continue  
        subj,rest = match.groups()
        match = re.search("NAP_1002_T2-rpac.fif", file)
        if not match:
            continue
        k = 0
        if "frontal" in file:
            k = 1
        elif "parietal" in file:
            k = 2
        if int(subj) not in [1001,1002,1003,1004,1005,1006,1008,1011,1012,1013,1015,1020,1023,1036,1038,1042,1046,1054,1055,1056,1057,1059]:
            continue

        # sort out conditions and polarity
        if subdir == "sham":
            cond = "sham"
            polarity = "anodal" # ?!?
        elif subdir == "sotDCS_anod":
            cond = "SOstim"
            polarity = "anodal"
        elif subdir == "sotDCS_cat":
            cond = "SOstim"
            polarity = "cathodal"
        elif subdir == "tDCS": # ?!?
            cond = "stim"
            polarity = "tDCS"
        else:
            raise ValueError("Could not organise condition/polarity")
        # = gap_dict[subj][polarity]
        
        if [subj, cond] in exclude:
            continue
        if include and [subj, cond] not in include:
            continue

        print(f"Processing {subj} {cond}")

        # create dataframes for all results
        bp = spindles_yasa(join(proc_dir, subdir, file),subj,cond, with_so,k)

        bp = bp.assign(subject = [subj]*len(bp), condition = [cond]*len(bp), polarity = [polarity]*len(bp))

        if number_of_preprocessed_files == 0:
            all_bp = bp
        else:
            all_bp = pd.concat([all_bp, bp],axis=0)
        
        number_of_preprocessed_files += 1

#save dataframes
all_bp = all_bp[["subject","condition","polarity","area","SO","spindles","TotalAbsPow","FreqRes","Relative"]]
all_bp=all_bp.astype({"subject":"int"})
all_bp=all_bp.sort_values("subject")
all_bp.to_csv(f"/media/Linux6_Data/johnsm/T2_sham_bandpower_{file_name}.txt", sep = "\t", index = False)

        

