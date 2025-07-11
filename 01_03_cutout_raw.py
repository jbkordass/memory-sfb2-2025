#!/usr/bin/env python
# coding: utf-8

# ## Cutout/reorganize raw file and annotations
# 
# ~~cut away everything except the desired periods of time before and/or after stimulation~~
# 
# For now instead of cutting/cropping/etc. just merge raw and annotations again.
# 
# Reject segments that have been concatenated earlier by replacing the annotation description by one starting with "BAD".

# In[ ]:


import mne
from os import listdir
import re
from os.path import isdir, join


# In[ ]:


root_dir = "/media/Linux6_Data/DATA/SFB2"
proc_dir = join(root_dir, "proc") # working directory

ignordir = ["SO_epochs_NOdetr","SO_epochs_detr","figs"]
sub_dirs = [item for item in listdir(proc_dir) if isdir(join(proc_dir, item))and item not in ignordir]

overwrite = False

testing = False # for testing, only run once


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
        match = re.search("NAP_(?P<subj>\\d{4})_(?P<cond>T\\d{1})-rpa.fif", file)
        if not match:
            continue
        subj = match.group('subj')
        cond = match.group('cond')
        print(f"Processing {subj} {cond}")

        #if subj != "9056":
        #    continue

        # set the file name of the output .fif file and check if it already exists in proc_dir
        outfile =  f"NAP_{subj}_{cond}-rpac.fif"
        if outfile in proclist and not overwrite:
            print(f"{outfile} already exists in processing dir. Skipping...")
            continue

        number_of_preprocessed_files += 1

        # now do the actual processing step
        # ---------------------------------

        # load up raw and annotations
        try:
            raw = mne.io.Raw(join(proc_dir, subdir, f"NAP_{subj}_{cond}-rp.fif"), preload=True)
            annots = mne.read_annotations(join(proc_dir, subdir, file))

            # check if there is concatenation point in a post stim interval or in the toend interval - if so reject interval
            for annot in annots:
                if "Brainvision concatenation" in annot["description"]:
                    for idx, possibly_relevant_annot in enumerate(annots):
                        if "Pre_Stimulation" in possibly_relevant_annot or "Post_Stimulation" in possibly_relevant_annot["description"] or "ToEnd" in possibly_relevant_annot["description"]:
                            if possibly_relevant_annot["onset"] < annot["onset"] and possibly_relevant_annot["onset"] + possibly_relevant_annot["duration"] > annot["onset"]:
                                annots.rename({ possibly_relevant_annot["description"]: "BAD [Brainvision concatenation contained]" }) 
                                print(f"Rejecting interval {possibly_relevant_annot["description"]} due to concatenation point.")
                                
        except:
            print(f"Error loading raw/stim annotations for {subj} {cond}")
            continue
        raw.set_annotations(annots)

        raw.save(join(proc_dir, subdir, outfile), overwrite=overwrite)
        
        # TODO: No idea what that special case is, commented out for now
        # special cases
        # if subj == "1026" and cond == "anodal":
        #     raw.crop(tmax=4180)

        # # run through all the annotations, cutting out the pre or post stimulation ones (ignore stimulation itself)
        # raws = []
        # for annot in annots:
        #     print(annot)

        #     match = re.match("(.*)_Stimulation (\d.*)", annot["description"])
        #     if match:
        #         (stim_pos, stim_idx) = match.groups()
        #         if stim_pos == "Active":
        #             continue
        #         # get the onset and duration times
        #         begin, duration = annot["onset"], annot["duration"]
        #         end = begin + duration
        #         # in case the annotation goes beyond the end of the recording (shouldn't happen anyway)
        #         if end > raw.times[-1]:
        #             end = raw.times[-1]
        #         try:
        #             raws.append(raw.copy().crop(begin, end))
        #         except:
        #             pass

        # # if there were no pre/post stimulation periods
        # if len(raws) == 0:
        #     continue

        # print("now merging...")

        # # now merge into one file
        # raw_cut = raws[0]
        # raw_cut.append(raws[1:])
        # raw_cut.save(join(proc_dir, subdir, outfile), overwrite=overwrite)


# ### Helpers
# 
# Cells should carry the `no_export` tag that is removed from the output in the .ipynb to .py conversion script `_ipynb_to_py.sh`.
# 
# Some helpers to explore annotations:
