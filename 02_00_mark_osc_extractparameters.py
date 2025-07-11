#!/usr/bin/env python
# coding: utf-8

# ## Mark slow and delta oscillations in data and save as raw and epoched formats

# adjusted file: extracts additional SO Parameters and writes them into a txt file

# In[ ]:


import mne
import math
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


# In[ ]:


root_dir = "/media/Linux6_Data/DATA/SFB2"
proc_dir = join(root_dir, "proc") # working directory
ignordir = ["SO_epochs_NOdetr","SO_epochs_detr","figs"]
sub_dirs = [item for item in listdir(proc_dir) if isdir(join(proc_dir, item))and item not in ignordir]

overwrite = True

testing = False # for testing, only run once

# Define ROIs and alternatives, one Channel or a List is possible:
#
# e.g.
# chan_groups_A = {"frontal":["Fz", "FC1", "FC2"], "parietal":["Cz", "CP1", "CP2"]}
# ...  
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

amp_percentile = 65
min_samples = 10
# Define SO Freq of interest
# Also Lists are possible
#minmax_freqs = [(0.16, 1.25), (0.75, 4.25)]
#minmax_times = [(0.8, 2), (0.25, 1)]
minmax_freqs = [(0.16, 1.25)]
minmax_times = [(0.8, 2)]
# osc_types = ["SO", "DELTA"]
osc_types = ["SO"]
#include = [
#            ["1002", "T3"],
#            ["1023", "T4"],
#            ["1059", "T1"]
#]
include = []
skipped = {"no_osc":[], "few_osc":[], "chan":[], "ROI":[]}


# In[ ]:


class OscEvent():
    # organising class for oscillatory events
    def __init__(self, start_time, end_time, middle, peak_time, peak_amp, trough_time,
                 trough_amp, slope, ptp, ptp_time, pos_duration, neg_duration, subj, cond, polarity, k):
        self.start_time = start_time
        self.end_time = end_time
        self.peak_time = peak_time
        self.peak_amp = peak_amp
        self.trough_time = trough_time
        self.trough_amp = trough_amp
        self.event_id = None
        self.event_annot = None
        #added new characteristics
        self.middle = middle
        self.slope=slope
        self.ptp=ptp
        self.ptp_time=ptp_time
        self.pos_duration=pos_duration
        self.neg_duration=neg_duration
        self.cond = cond
        self.subj = subj
        self.polarity = polarity
        self.k = k

    def __repr__(self):
        return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.subj, self.cond, self.polarity, self.k, self.start_time,
                                                             self.end_time, self.peak_time,
                                                             self.peak_amp, self.trough_time,
                                                             self.trough_amp, self.slope, self.ptp_time, self.ptp, self.neg_duration, self.pos_duration)

def check_trough_annot(desc):
    # helper function for marking troughs of oscillations
    event = None
    if "Trough" in desc and "ToEnd" not in desc:    
        event = int(desc[-1]) # Number of Post_stimulation X -> desc[-1] = X
    if "Trough" in desc and "ToEnd" in desc:    
        event = 99  
    return event

def get_annotation(annotations, time):
    # does a time period reside in a Post stim annotation?
    period = None
    for annot in annotations:
        if "Post" not in annot["description"]:
            continue
        begin = annot["onset"]
        end = begin + annot["duration"]
        if time > begin and time < end:
            period = annot["description"]
    return period

def osc_peaktroughs(osc_events):
    # get peaks and troughs of an OscEvent instance
    peaks = []
    troughs = []
    
    for oe in osc_events:
        peaks.append(oe.peak_amp)
        troughs.append(oe.trough_amp)
        
    peaks, troughs = np.array(peaks), np.array(troughs)
    return peaks, troughs

def mark_osc_amp(osc_events, amp_thresh, chan_name, mm_times, osc_type,
                 raw_inst=None):
    # 
    output=open("/media/Linux6_Data/DATA/SFB2/so_parameters.txt","a")
    osc_idx = 0
    for oe in osc_events:
        if raw_inst is not None:
            event_annot = get_annotation(raw_inst.annotations,
                                         oe.start_time)
            if event_annot is None:
                continue
        else:
            event_annot = None
        pt_time_diff = oe.trough_time - oe.peak_time
        time_diff = oe.end_time - oe.start_time
        pt_amp_diff = oe.peak_amp - oe.trough_amp
        if pt_amp_diff > amp_thresh and mm_times[0] < time_diff < mm_times[1]:
            oe.event_id = "{} {} {}".format(chan_name, osc_type, osc_idx)
            oe.event_annot = event_annot
            osc_idx += 1
            print(oe, file=output)
    output.close()


def is_timepoint_in_intervals(tp, onsets, durations):
    for onset, duration in zip(onsets, durations):
        if onset <= tp <= (onset + duration):
            return True
    return False


# In[ ]:


number_of_preprocessed_files = 0

output=open("/media/Linux6_Data/DATA/SFB2/so_parameters.txt","w")
output.write("subject\tcondition\tpolarity\tk\tstart\tend\tpeak\tpeak_amp\ttrough\ttrough_amp\tslope\tptp_time\tptp\tnegative_duration\tpositive_duration")
output.close()

for subdir in sub_dirs:
    if subdir == "tDCS":
        continue

    print(f"{subdir}".center(80, '-'))
    proclist = listdir(join(proc_dir, subdir)) 
    #save_dir = join("/media/Linux6_Data/johnsm/outputs",subdir)
    save_dir = join(proc_dir,subdir)
    for file in proclist:

        # if we are testing, only apply step to one file
        if testing and number_of_preprocessed_files > 0:
            continue

        # find out subject id (4 digits) and condition (T1, T2, T3, T4) from file name
        match = re.search("NAP_(?P<subj>\\d{4})_(?P<cond>T\\d{1})-rpacbi.fif", file)
        if not match:
            continue
        subj = match.group('subj')
        cond = match.group('cond')

        if include and [subj, cond] not in include:
            continue

        print(f"Processing {subj} {cond}")

        # set the file name of the output .fif file and check if it already exists in proc_dir
        outfile =  f"NAP_{subj}_{cond}-rpaco.fif"
        if outfile in proclist and not overwrite:
            print(f"{outfile} already exists in processing dir. Skipping...")
            continue

        number_of_preprocessed_files += 1

        # now do the actual processing step
        # ---------------------------------
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
                    skipped["chan"].append("{} {} {} {}".format(subj, cond ,k , v))
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
            skipped["ROI"].append("{} {}".format(subj, cond))
            continue
  
        # sort out conditions and polarity
        if subdir == "sham":
            cond = "sham"
            polarity = "anodal"  
        elif subdir == "sotDCS_anod":
            cond = "SOstim"           
            polarity = "anodal"
        elif subdir == "sotDCS_cat":
            cond = "SOstim"      
            polarity = "cathodal"
        elif subdir == "tDCS": 
            cond = "tDCSstim"      
            polarity = "anodal"   
        else:
            raise ValueError("Could not organise condition/polarity")
        # = gap_dict[subj][polarity]
        

        for minmax_freq, minmax_time, osc_type in zip(minmax_freqs, minmax_times, osc_types):
            raw_work = raw.copy()
            raw_work.filter(l_freq=minmax_freq[0], h_freq=minmax_freq[1])
            first_time = raw_work.first_samp / raw_work.info["sfreq"]

            # zero crossings
            for k in raw.ch_names:
                df_dict = {"Subj":[],"Cond":[],"Index":[], "ROI":[], "Polarity":[],
                        "OscType":[], "OscLen":[], "OscFreq":[]}
                pick_ind = mne.pick_channels(raw_work.ch_names, include=[k])
                

                signal = raw_work.get_data()[pick_ind,].squeeze()

                # need to add infinitesimals to zeros to prevent weird x-crossing bugs
                for null_idx in list(np.where(signal==0)[0]):
                    if null_idx:
                        signal[null_idx] = 1e-16*np.sign(signal[null_idx-1])
                    else:
                        signal[null_idx] = 1e-16*np.sign(signal[null_idx+1])

                zero_x_inds = (np.where((signal[:-1] * signal[1:]) < 0)[0]) + 1
                # cycle through negative crossings
                neg_x0_ind = 1 if signal[0] < 0 else 2
                osc_events = []
                Bad_onsets = [onset for onset, desc in zip(raw_work.annotations.onset, raw_work.annotations.description) if "BAD" in desc]
                Bad_durations = [duration for duration, desc in zip(raw_work.annotations.duration, raw_work.annotations.description) if "BAD" in desc]
                for zx_ind in range(neg_x0_ind, len(zero_x_inds)-2, 2):
                    idx0 = zero_x_inds[zx_ind]
                    idx1 = zero_x_inds[zx_ind+1]
                    idx2 = zero_x_inds[zx_ind+2]
                    if (idx1 - idx0) < min_samples or (idx2 - idx1) < min_samples:
                        continue
                    time0 = raw_work.first_time + raw_work.times[idx0]
                    time1 = raw_work.first_time + raw_work.times[idx2]
                    midcrossing = raw_work.first_time + raw_work.times[idx1]
                    peak_time_idx = np.min(find_peaks(signal[idx1:idx2])[0]) + idx1
                    trough_time_idx = np.argmin(signal[idx0:idx1]) + idx0
                    peak_amp, trough_amp = signal[peak_time_idx], signal[trough_time_idx]
                    peak_time = raw_work.first_time + raw_work.times[peak_time_idx]
                    trough_time = raw_work.first_time + raw_work.times[trough_time_idx]
                    #add additional characteristics
                    ptp = abs(trough_amp)+peak_amp
                    ptp_time=peak_time-trough_time
                    pos_duration=raw_work.times[idx2]-raw_work.times[idx1]
                    neg_duration=raw_work.times[idx1]-raw_work.times[idx0]
                    slope=ptp/ptp_time
                    # Reject SO-candidates if first zerocrossing time is in a Bad Timespan
                    if not is_timepoint_in_intervals(time0, Bad_onsets, Bad_durations):
                        osc_events.append(OscEvent(time0, time1, midcrossing, peak_time,
                                                peak_amp, trough_time, trough_amp, slope, ptp, ptp_time, pos_duration, neg_duration, subj, cond, polarity, k))

                # Reject SO-candidates if length is to short or to long: minTime < SOlength <maxTime
                osc_events = [oe for oe in osc_events if (oe.end_time-oe.start_time)>minmax_time[0] and 
                            (oe.end_time-oe.start_time)<minmax_time[1]]
                
                peaks, troughs = osc_peaktroughs(osc_events)
                amps = peaks - troughs
                amp_thresh = np.percentile(amps, amp_percentile)

                print("Amp-thresh: " + str(amp_thresh))

                mark_osc_amp(osc_events, amp_thresh, k, minmax_time, osc_type,
                            raw_inst=raw_work)
                marked_oe = [oe for oe in osc_events if oe.event_id is not None]
                if len(marked_oe):
                    for moe_idx, moe in enumerate(marked_oe):
                        if moe_idx == 0:
                            new_annots = mne.Annotations(moe.start_time,
                                                            moe.end_time-moe.start_time,
                                                            "{} {}".format(moe.event_id, moe.event_annot),
                                                            orig_time=raw_work.annotations.orig_time)
                        else:
                            new_annots.append(moe.start_time, moe.end_time-moe.start_time,
                                                "{} {}".format(moe.event_id, moe.event_annot))
                        new_annots.append(moe.trough_time, 0,
                                            "Trough {} {}".format(moe.event_id, moe.event_annot))
                        new_annots.append(moe.peak_time, 0,
                                            "Peak {} {}".format(moe.event_id, moe.event_annot))
                        # add annotation for positive half wave of SO
                        new_annots.append(moe.middle, moe.end_time-moe.middle,
                                            "PosHalfWave {} {}".format(moe.event_id, moe.event_annot))
                    new_annots.save(join(save_dir,
                                        f"osc_NAP_{subj}_{cond}_{k}_{osc_type}_{polarity}-annot.fif"),
                                    overwrite=True)
                    raw.set_annotations(new_annots)
                else:
                    skipped["no_osc"].append("{} {} {} {} {}".format(subj, cond, k, osc_type, polarity))
                    print("\nNo oscillations found. Skipping.\n")
                    continue

                events = mne.events_from_annotations(raw, check_trough_annot)
                
                for event_idx, event in enumerate(events[0][:,-1]):
                    eve = event.copy()
                    df_dict["Index"].append(int(eve))
                    df_dict["Subj"].append(subj)
                    df_dict["Cond"].append(cond)
                    df_dict["Polarity"].append(polarity)
                    #df_dict["Gap"].append(gap)
                    df_dict["ROI"].append(k)
                    df_dict["OscType"].append(osc_type)
                    df_dict["OscLen"].append(marked_oe[event_idx].end_time - marked_oe[event_idx].start_time)
                    df_dict["OscFreq"].append(1/df_dict["OscLen"][-1])

                df = pd.DataFrame.from_dict(df_dict)
                epo = mne.Epochs(raw, events[0], tmin=-2.5, tmax=2.5, detrend=1,
                                baseline=None, metadata=df, event_repeated="drop",
                                reject={"eeg":5e-4}).load_data()
                # Checking metadata
                assert isinstance(epo.metadata, pd.DataFrame)
                print("######################################")
                print(epo.metadata)
                print("######################################")
                #if len(epo) < 25:
                #    skipped["few_osc"].append("{} {} {} {} {}".format(subj, cond, k, osc_type, polarity))
                #    continue
                raw.save(join(save_dir, f"osc_NAP_{subj}_{cond}_{k}_{osc_type}_{polarity}-raw.fif"),
                        overwrite=True)
                epo.save(join(save_dir, f"osc_NAP_{subj}_{cond}_{k}_{osc_type}_{polarity}-epo.fif"),
                        overwrite=True)

    # with open(join(proc_dir, "skipped_record.pickle"), "wb") as f:
    #     pickle.dump(skipped, f)

    print(f"Skipped: {skipped}")


# In[ ]:





# In[ ]:




