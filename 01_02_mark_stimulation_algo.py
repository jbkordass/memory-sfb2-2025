#!/usr/bin/env python
# coding: utf-8

# ## Mark stimulation intervals algorithmically
# 
# Tries to detect stimulation intervals in files with sotDCS stimulation. Does not rely on triggers/annotations.

# In[ ]:


import mne
from os import listdir,getcwd
from os.path import isdir, join
import re
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
plt.ion()

from scipy.signal import find_peaks


# ## Setup and define thresholds
# 
# These seem to be experimentally determined.
# 
# ### Note on NAP_1056_T1 (stimulation not working on both sides)
# - Stimulation on the left seems to have worked, but also has lower amplitudes compared to the right
# - Choosing the following parameters does in fact detect the stimulation period in question
#   ```
#   tfr_lower_thresh = 1e-4 # "reduce" detection threshold
#   picks = ["Fz","AFz", "Cz", "Fp2","FC2"] # only pick right electrodes
#   ```
# - using also RHS picks (even with modified threshold) will not lead to detection of the failed stimulation
# - raising the `tfr_lower_thresh` threshold, also extends the detected stimulation periods (as we would expect)
# - *Technical note: to do the comparison it seems helpful to run the script twice, i.e. use files where `cutout_raw` already ran by setting the input file ending to `rpac`*

# In[ ]:


root_dir = str("/media/Linux6_Data/DATA/SFB2")
proc_dir = join(root_dir, "proc") # working directory

sub_dirs = [item for item in listdir(proc_dir) if isdir(join(proc_dir, item))]

n_jobs = 24

overwrite = True

testing = True # for testing, if True, only run once

# include or exclude, ignored if empty
exclude = [
    #["1003", "T2"],
    #["1064", "T2"]     # wrong tDCS intervals detected
]
skipDir = []       
include = [["1002","T2"]
    #["1074", "T2"],
    ###["1002", "T1"], #  PostStimtoEnd fehlt, weil EEG nach letzter Stim endet
    ###["1015", "T3"], #  PostStimtoEnd sehr kleiner Abschnitt, weil EEG kurz nach der Stim 6 endet
    ###["1019", "T1"], #  erledigt:  Stim1 + Stim 2 zum Teil markiert; Stim3 nicht detektiert, Stim4 als Stim3 detektiert, Stim5 nicht detektiert, Stim6 als Stim4 detektiert
    ###["1023", "T1"], #  Stimulation 14 vorhanden aber kein Interval + PostStimtoEnd, weil EEG zu Ende
    ###["1042", "T3"], #  PostStimtoEnd Markierung tritt nicht direkt nach der letzten Stim, sondern später
    ###["1074", "T2"], #  Bis zu 4. Stim richtig markiert; nach der 4ten wurde alles als Response/R128 markiert - diese wurden aber entfernt ; Falsch markierte Intervalle: Stimulation 5-10 + entsprechende Bad_Ative Stimulation + Bad_Buffer_Stimulation Marker; kein PostStimtoEnd Marker auch unter Annotation Liste nicht
    ###["1004", "T4"], #  kein PostStimtoEnd, weil EEG bei letztem Interval endet
    ###["1023", "T4"], #  erledigt  PreStimulation, PostStim 0 und 1 (plus BAD_Buffer und BAD_active Marker) falsch markiert ; Stimulation 0 nicht detektiert und entfernt // die ersten zwei Stimulation passieren hintereinander - es ist nicht klar welcher der 1 Stim zwischen den beiden ist, weil es gibt kein Interval dazwischen // Diese wurde als PostStim2 etc. entfernt weil falsch markiert; fängt bei Stim3 an - sollte Stim1 sein // Stim 9 + Stim 12 + 13 nicht markiert bzw. wenn man ab Stim 3 zählt dann Stim 11,14,15
    ###["1036", "T4"], #  PostStim fängt nicht direkt nach Stim_Interval an
    ###["1042", "T4"], #  ist OK so:  PostStimtoEnd fängt später an, nicht direkt nach der letzten Stim/ Stim4 + Stim 6 nicht detektiert - stattdessen BAD Brainvision concatenation contained Marker - nicht entfernt
    ###["1056", "T4"], #  Stim 12 ist vorhanden aber nicht der Intervall, weil EEG endet + keine PostStimtoEnd
    ###["1042", "T1"], #  PostStimtoEnd fängt nicht direkt nach dem interval an
    ###["1047", "T2"], #  PostStimtoEnd fängt später an, nicht direkt nach letztem marker
    ###["1056", "T4"], #  Stim 12 ist vorhanden aber nicht der Intervall, weil EEG endet + kein PostStimtoEnd
    ###["1004", "T2"], #  erledigt!  12 Stims, Sollten aber 13 sein -- Comment/10 Tim a + Comment not usuable marker könnte der 13.Stim sein (nicht entfernt); PostStimtoEnd sehr kleiner Abschnitt, weil EEG endet
    ###["1013", "T3"], #  PostStimtoEnd kleiner abschnitt, weil EEG endet
    ###["1038", "T1"], #  Ist OK so
    ###["1042", "T1"], #  Ist OK so
    ###["107", "T1"]   #  Ist OK so
]
# BAD Channel
# ---------
# Define a List of Bad channel per session manually. Later

ses_bad_ch = {#
    #"1019_T1" : ["AFz", "FC1", "CP1", "Fz"],
    #"1023_T4" : ["AFz", "FC1", "Cz"],
    #"1042_T3" : ["Cz", "Fz"]
}


# Durations
# ---------

# buffer times before and after stimulation detection timepoints
pre_stim_buffer = 20    #pre_stim window ends 20 sec before stimulation onset
post_stim_buffer = 15   #post_stim window starts 30 (jevri) sec after stimunlation end

# analysis window durations
analy_duration = 60 # duration of the (pre/poststim) analysis window
last_analy_duration = [] # duration of the last analysis window
between_duration = 60

# sham stimulation duration, asserted in detecting sham stimulations
sham_stim_duration = 60 # in seconds


# sotDCS detection algorithm parameters
# -------------------------------------

post_only = True # if True, only mark the first prestim interval (otherwise mark prestim and poststim for each stimulation)

reject_by_annotation = False # Default is "True". Changed to "False" to only exclude Bad-Channels and no data rejections by annotations named "bad_*"  -> only for "*-rpi.fif" files.

# some threshold parameters to determine when stimulation actually happens
tfr_upper_thresh_range = list(np.linspace(0.001,0.01,100)) # go from 0.001 to 0.01 in 100 steps
#tfr_upper_thresh_range = list(np.linspace(0.001,0.006,100)) # go from 0.001 to 0.01 in 100 steps
#tfr_upper_thresh_range = list(np.linspace(1e-5,0.0018,100)) # go from 0.001 to 0.005 in 100 steps   -> decreased range to grab more stims
tfr_lower_thresh = 1e-6 

epolen = 10 # length in seconds of epochs the data is cut into for tfr analysis
min_stim_duration = 25 # min duration in seconds for a stimulation interval to be considered valid
#min_stim_duration = 10
#picks = ["AFz","Fp1","Fp2"]
picks = ["Fz","AFz", "Cz", "Fp2","FC2", "Fp1", "FC1"] # channels used to determine if stimulation has occured
# dur_dict = {344:"5m", 165:"2m", 73:"30s"}


# tDCS algorithm parameters
# -------------------------
peak_width = 1000 # 1000 default, 50 if HP Filter (0.1Hz) and LP Filter (100Hz) in prep
peak_height = 3 # 3, height of the peaks in terms of standard deviations of the signal


# ## Notes on Problems with detection
# - the detection algorithm used a bunch of assumptions on the data file
# - it will necessarily run into problems in the following cases:
#   + if the stimulation frequency varies (because some stimulations did not work/worked differently through stimulations)
#     - here 
#   + if the stimulation intervals are varying in length!

# In[ ]:


def mod_pks(expcs):
    newpicks = picks.copy()
    #print(new_picks)
    for pks in picks:
        if pks in expcs:
            newpicks.remove(pks)
    return newpicks


# In[ ]:


def detect_sotDCS_stimulation(filename, outfilename, excludeCH):
    """
    Jevris sotDCS detection algorithm

    Rough steps: 
    1. Compute power spectrum and take the mean over picked channels, use this to construct a function tfr_aschan
       that assigns to every timepoint the mean tfr value
    2. Iterate through upper threshold range beginning with the highest values
    2.1. Iterate through the crossings of the threshold tfr_upper_thresh
    2.1.1 Find first crossing of the lower threshold after the minimum stimulation duration
    2.1.2 Take this as a "stimulation interval" and coninue with the next crossing of the upper thresh after the end of it
    2.2. Compute the standard deviation of the durations of these "stimulation intervals" found
    2.3. If the standard deviation is lower than the current "winner" save and continue with next upper threshold in range
    3. Assume "winner" is the best threshold and save the annotations
    
    Saves results in as an annotation to the file 'outfilename'.
    """

    ur_raw = mne.io.Raw(filename, preload=True)
    raw = ur_raw.copy()
        
    #raw.info["bads"] = excludeCH
    mod_picks = mod_pks(excludeCH)   
    
    # compute power spectrum to determine the frequency with the highest power ?! 
    # I guess the idea is that this should be the stimulation
    
    spectrum = raw.compute_psd(method="multitaper", fmax=2, picks=mod_picks, n_jobs=n_jobs,reject_by_annotation=reject_by_annotation)
    #spectrum = raw.compute_psd(method="welch", fmax=2, picks=picks, n_jobs=n_jobs)  # 'welch'(s) Method is working with "reject_by_annotation=True" (default)
    psd = spectrum.get_data().mean(axis=0)
    fmax = spectrum.freqs[np.argmax(psd)]
    print(f"Asserted stimulation frequency is {fmax}")
    # fmax should be sth like 0.75
    # maybe it would be a good idea to throw an exception if it is not
    
    # now cut data into fixed length epochs (epolen specifies the number of seconds, usually 10)
    # (note that this would in principle already exclude data annotated with "BAD"!)
    epochs = mne.make_fixed_length_epochs(raw, duration=epolen, reject_by_annotation=reject_by_annotation)
    tfr_epochs = tfr_morlet(epochs, 
                            n_cycles = 1,
                            freqs = [fmax],
                            average = False,
                            return_itc = False,
                            picks = mod_picks, 
                            n_jobs = n_jobs)
    # starting from MNE version 1.7 better use
    # tfr_epochs = epochs.compute_tfr(method = "morlet", 
    #                         freqs = [fmax],
    #                         average = False,
    #                         return_itc = False,
    #                         picks = picks, 
    #                         n_jobs = n_jobs)

    # run through all the epochs and find the mean tfr values over all channels
    # then concatenate them to a vector of length the number of timepoints in the entire recording (tfr)
    tfr = np.zeros(0)
    for epo_tfr in tfr_epochs.__iter__():
        # take the mean over channels (picks) at every timepoint of the epoch 
        # i.e. np.mean(epo_tfr[:,0,],axis=0) is a vector of length the number of timepoints (usally 10s * 200Hz)
        tfr = np.concatenate((tfr, np.mean(epo_tfr[:,0,],axis=0)))
    # now create a dictionary that assigns to every timepoint the mean tfr value
    tfr_aschan = np.zeros(len(raw))
    tfr_aschan[:len(tfr)] = tfr # possibly tfr is shorter than the recording

    # iterate through the threshold range specified above (usually 100 steps)
    # as a "convergence measure" we take the standard deviation of the durations of the stimulation periods detected
    # Remark: probably works best if there are many stimulations in the data!
    winner_std = np.inf
    for tfr_upper_thresh in tfr_upper_thresh_range:

        these_annotations = raw.annotations.copy()

        # check where the tfr values are over the upper treshold
        # 
        # this is actually a simple trick to find points in which the threshold is crossed
        # the first line assings 0.5 to all points above the threshold and -0.5 to all points below the threshold
        # the second line multiplies the values of the first line at timepoint t with the values of the first line at timepoint t+1
        # ⇒ if the threshold is crossed, the product is negative
        tfr_over_upper_thresh = (tfr_aschan > tfr_upper_thresh).astype(float) - 0.5 # > 0 if above threshold, < 0 if below
        tfr_upper_tresh_cross = tfr_over_upper_thresh[:-1] * tfr_over_upper_thresh[1:]
        tfr_upper_tresh_cross = np.concatenate((np.zeros(1),tfr_upper_tresh_cross))

        # and under the lower threshold
        tfr_under_lower_thresh = (tfr_aschan < tfr_lower_thresh).astype(float) - 0.5 # > 0 if below threshold, < 0 if above
        tfr_lower_thresh_cross = tfr_under_lower_thresh[:-1] * tfr_under_lower_thresh[1:]
        tfr_lower_thresh_cross = np.concatenate((np.zeros(1),tfr_lower_thresh_cross))

        # as described above tfr_upper_tresh_cross < 0 are the points where the threshold is crossed
        tfr_lower_thresh_cross_idcs = np.where(tfr_lower_thresh_cross < 0)[0]

        # if no crossings found, move on 
        # Remark: no idea why in previous versions this checked twice...
        if len(np.where(tfr_upper_tresh_cross < 0)[0]) == 0:
            continue

        # iterate through points where the upper threshold is crossed, i.e. where we expect a stimulation to begin at
        earliest_idx = 0 # earliest index to start searching for crossings of the upper threshold
        stim_number = 0 # current stimulation number
        for cross_idx in np.nditer(np.where(tfr_upper_tresh_cross < 0)[0]):

            # if the crossing is before the earliest index (e.g. because found a stimulation there already), skip
            if cross_idx < earliest_idx:
                continue

            # min_stim_duration is the minimum number of seconds for a stimulation period 
            # so we compute the index of the end of the minimum stimulation duration
            min_stim_duration_end_idx = cross_idx + int(np.round(min_stim_duration * raw.info["sfreq"]))

            # do an end of recording check, the length here is just the same as len(raw)
            if min_stim_duration_end_idx > len(tfr_under_lower_thresh):
                min_stim_duration_end_idx = len(tfr_under_lower_thresh) - 1
            # now check if the value at min_stim_duration_end_idx is below the lower threshold
            # the ideas is that the tfr values beeing already low means we have a short peak or something else that 
            # causes tfr to be high, but not a stimulation
            if tfr_under_lower_thresh[min_stim_duration_end_idx] > 0: # false alarm; do not mark
                earliest_idx = min_stim_duration_end_idx
                continue
            
            # calulate times for the currently asserted stimulation
            begin = raw.times[cross_idx] - pre_stim_buffer
            # find the first crossing of the lower threshold after the minimum duration
            first_reasonable_lower_thresh_cross_idx = tfr_lower_thresh_cross_idcs[tfr_lower_thresh_cross_idcs > min_stim_duration_end_idx][0]
            end = raw.times[first_reasonable_lower_thresh_cross_idx]
            duration = end - begin

            # for the first stimulation set durations for pre and post intervals according to variables set above
            if stim_number == 0:
                pre_dur = analy_duration
                post_dur = between_duration
            else:
                pre_dur = between_duration
                post_dur = between_duration
            
            these_annotations.append(begin, duration, "BAD_Active_Stimulation {}".format(stim_number))
            these_annotations.append(end, post_stim_buffer, "BAD_Buffer_Stimulation {}".format(stim_number))
            # decide if prestim intervals should be annotated for each or only for the first stimulation
            if not post_only or not stim_number:
                these_annotations.append(begin - pre_dur, pre_dur, "Pre_Stimulation {}".format(stim_number))
            these_annotations.append(begin + duration + post_stim_buffer, post_dur, "Post_Stimulation {}".format(stim_number))
            stim_number += 1

            # start searching for next crossing only after the end of the "stimulation" found
            earliest_idx = first_reasonable_lower_thresh_cross_idx

        # assess uniformity
        # calculate the standard deviation of the durations of the stimulation periods
        durations = []
        for annot in these_annotations:
            if "Active" in annot["description"]:
                durations.append(annot["duration"])
        dur_std = np.array(durations).std()

        # if the standard deviation is lower than the current winner, save the current annotations
        if dur_std < winner_std and dur_std != 0.:
            winner_annot = these_annotations.copy()
            winner_std =  dur_std
            winner_id = tfr_upper_thresh
            winner_stim_idx = stim_number
            winner_durations = durations.copy()
    
    # last post-stimulation period should handled differently
    last_annot = winner_annot[-1].copy()

    # one way would be to remove the last winner annotation and then add a new one that extends to the end of the recording
    # winner_annot.delete(-1)
    # winner_annot.append(last_annot["onset"], analy_duration, last_annot["description"])
 
    # instead we decide to just add a new annotation that extends to the end of the recording
    last_analy_duration = raw._last_time - (last_annot["onset"] + analy_duration)
    winner_annot.append(last_annot["onset"] + analy_duration, last_analy_duration, "Post_Stimulation_ToEnd")

    print(f"Threshold of {winner_id} was optimal.\nDurations:")
    print(winner_durations)
    print(f"Standard deviation (of durations):{winner_std}")
    print(f"Found {winner_stim_idx} stimulations.")

    # if the standard deviation is too high, we might want to check the results
    # if winner_std > 2:
    #   breakpoint()

    # save the annotations
    
    raw.set_annotations(winner_annot)
    winner_annot.save(outfilename, overwrite=overwrite)

    # if we are testing, draw a diagnostic plot of the thresholds and plot with mne
    if testing:
        import matplotlib.pyplot as plt
        get_ipython().run_line_magic('matplotlib', 'qt')
        fig = plt.figure(figsize=(50, 10))
        plt.plot(range(0, len(tfr)), tfr, lw= 1, color = 'slategrey')
        # add a line to the plot on the y-axis at the lower threshold value
        plt.axhline(y = tfr_lower_thresh, color = 'r', linestyle = '--')
        # add a line for the max in the upper threshold range
        plt.axhline(y = tfr_upper_thresh_range[np.argmax(tfr_upper_thresh_range)], color = 'g', linestyle = '--')
        # and for the min
        plt.axhline(y = tfr_upper_thresh_range[np.argmin(tfr_upper_thresh_range)], color = 'g', linestyle = '--')
        # add a line for the winner threshold
        plt.axhline(y = winner_id, color = 'b', linestyle = '--')
        # add a lengend what all the lines are
        plt.legend(["tfr", "lower threshold", "upper threshold range min", "upper threshold range max", "winner threshold"])
        # zoom in around the annotation with stim_idx
        stim_number = 0 # set to -1 to not zoom in at all, 0 would be the first stimulation
        for annot in winner_annot:
            if f"BAD_Active_Stimulation {stim_number}" in annot["description"]:
                plt.xlim(int(annot["onset"] * raw.info["sfreq"]), 
                         int((annot["onset"] + annot["duration"]) * raw.info["sfreq"]))


        # Add a dummy channel to the raw object (for plotting) that contains tfr_aschan
        info = mne.create_info(['Dummy:TFR'], raw.info['sfreq'], ['misc'])
        tfr_channel = np.ndarray((1, len(tfr_aschan)), buffer=np.array(tfr_aschan)*0.5) 
        tfr_raw = mne.io.RawArray(tfr_channel, info)
        raw.add_channels([tfr_raw], force_update_info=True)

        # plot (optional)
        raw.pick(picks + ['Dummy:TFR']).plot(block=True, scalings=dict(eeg=20e-3), duration=700)


# In[ ]:


def detect_tDCS_stimulation(filename, outfilename):
    """ 
    Algorithm to find tDCS stimulations in the data based on peak analysis in a channel summation function

    Currently does *not* use an optimazation procedure, possibly add that, if the performance is bad with less clear signals.

    Rough steps:
    1. Sum over all channels in picks to create a function x that has hight peaks where stimulations are
    2. Find peaks and assume inverse peaks at stimulation start and end
    3. Iterate through the peaks and find the first negative peak after the minimum stimulation duration
    4. Annotate the stimulation period and the pre and post stimulation periods
    """
    
    ur_raw = mne.io.Raw(filename, preload=True)
    raw = ur_raw.copy()
    these_annotations = raw.annotations.copy()

    # plot (optional)
    #raw.pick(picks).plot(block=True, scalings=dict(eeg=20e-3), duration=700)
    
    # compute sum over the channels in picks and call this function x
    x = np.zeros(len(raw))
    for pick in picks:
        x += raw.get_data(picks=pick).mean(axis=0)
    
    peaks, _ = find_peaks(x, height=np.mean(x) + peak_height*np.std(x), width=peak_width)
    neg_peaks, _ = find_peaks(-x, height=np.mean(x) + peak_height*np.std(x), width=peak_width)
    #print(neg_peaks)
    earliest_idx = 0 # earliest index to start searching for crossings of the upper threshold
    stim_number = 0 # current stimulation number
    for peak_index in peaks:

        # if the crossing is before the earliest index (e.g. because found a stimulation there already), skip
        if peak_index < earliest_idx:
            continue

        # after first_stim find first element in peaks_100
        stim_start_index = peak_index
        #print(stim_start_index)
        # find next element in neg_peaks_100 that is at least min_stim_duration away
        min_stim_duration_end_idx = stim_start_index + int(np.round(min_stim_duration * raw.info["sfreq"]))
        
        #print(min_stim_duration_end_idx)
        # do an end of recording check, the length here is just the same as len(raw)
        #print(np.where(neg_peaks > min_stim_duration_end_idx))
        if min_stim_duration_end_idx > len(raw):
            stim_end_index = len(raw) - 1
        else:
            stim_end_index = neg_peaks[np.where(neg_peaks > min_stim_duration_end_idx)[0][0]]
    
        # calulate times for the currently asserted stimulation
        begin = raw.times[stim_start_index] - pre_stim_buffer
        end = raw.times[stim_end_index]
        duration = end - begin

        # for the first stimulation set durations for pre and post intervals according to variables set above
        if stim_number == 0:
            pre_dur = analy_duration
            post_dur = between_duration
        else:
            pre_dur = between_duration
            post_dur = between_duration
    
        these_annotations.append(begin, duration, "BAD_Active_Stimulation {}".format(stim_number))
        these_annotations.append(end, post_stim_buffer, "BAD_Buffer_Stimulation {}".format(stim_number))
        # decide if prestim intervals should be annotated for each or only for the first stimulation
        if not post_only or not stim_number:
            these_annotations.append(begin - pre_dur, pre_dur, "Pre_Stimulation {}".format(stim_number))
        these_annotations.append(begin + duration + post_stim_buffer, post_dur, "Post_Stimulation {}".format(stim_number))
        stim_number += 1

        # start searching for next crossing only after the end of the "stimulation" found
        earliest_idx = stim_end_index

    # assess uniformity
    # calculate the standard deviation of the durations of the stimulation periods
    durations = []
    for annot in these_annotations:
        if "Active" in annot["description"]:
            durations.append(annot["duration"])
    dur_std = np.array(durations).std()

    # currently there is no optimization, so the first try yields the winner...
    winner_annot = these_annotations.copy()
    winner_std =  dur_std
    winner_id = -1
    winner_stim_idx = stim_number
    winner_durations = durations.copy()
    
    # last post-stimulation period should handled differently
    last_annot = winner_annot[-1].copy()

    # one way would be to remove the last winner annotation and then add a new one that extends to the end of the recording
    # winner_annot.delete(-1)
    # winner_annot.append(last_annot["onset"], analy_duration, last_annot["description"])
 
    # instead we decide to just add a new annotation that extends to the end of the recording
    last_analy_duration = raw._last_time - (last_annot["onset"] + analy_duration)
    winner_annot.append(last_annot["onset"] + analy_duration, last_analy_duration, "Post_Stimulation_ToEnd")

    # print(f"Threshold of {winner_id} was optimal.\nDurations:")
    print(winner_durations)
    print(f"Standard deviation (of durations):{winner_std}")
    print(f"Found {winner_stim_idx} stimulations.")

    # if the standard deviation is too high, we might want to check the results
    # if winner_std > 2:
    #   breakpoint()

    # save the annotations
    raw.set_annotations(winner_annot)
    winner_annot.save(outfilename, overwrite=overwrite)

    # if we are testing, draw a diagnostic plot of the thresholds and plot with mne
    if testing:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(50, 10))
        plt.plot(range(0, len(x)), x, lw= 1, color = 'slategrey')
        stim_number = 0 # set to -1 to not zoom in at all, 0 would be the first stimulation
        for annot in winner_annot:
            if f"BAD_Active_Stimulation {stim_number}" in annot["description"]:
                plt.xlim(int(annot["onset"] * raw.info["sfreq"]), 
                         int((annot["onset"] + annot["duration"]) * raw.info["sfreq"]))


        # Add a dummy channel to the raw object (for plotting) that contains tfr_aschan
        info = mne.create_info(['Dummy:Pick Summation'], raw.info['sfreq'], ['misc'])
        dummy_channel = np.ndarray((1, len(x)), buffer=np.array(x)*0.5) 
        dummy_raw = mne.io.RawArray(dummy_channel, info)
        raw.add_channels([dummy_raw], force_update_info=True)

        # plot (optional)
        raw.pick(picks + ['Dummy:Pick Summation']).plot(block=True, scalings=dict(eeg=20e-3), duration=700)
        


# 

# In[ ]:


def detect_sham_stimulation(filename, outfilename):
    """ 
    Detect sham stimulation based on triggers
    """
    raw = mne.io.Raw(join(proc_dir, filename), preload=True)

    # we want to add new annotations to the old ones, so begin with these
    stim_annots = raw.annotations.copy()
    # stim_annots = mne.Annotations([], [], [])

    # run through the annotations and try to find begin/end of sham stimulation triggers
    # we can assume the annotations are sorted by onset
    sham_stims = {}
    counted_stim_number = 1
    for annot in raw.annotations:
        # find the strings "comment", "stim", "anf" or "end" using positive lookahead (?=)
        # also try to identify the stimulation number (though we rather use our own counter...)
        match = re.match("(?=.*comment)(?=.*(stim|st))(?=.*(?P<type>begin|end))(?=\\D*(?P<number>(\\d+))).+", annot["description"], re.IGNORECASE)
        if match:
            stim_number = int(match.group("number"))
            if not counted_stim_number in sham_stims.keys():
                sham_stims[counted_stim_number] = {}

            if match.group("type") in [ "begin" ]:
                if counted_stim_number != stim_number:
                    # should not happen, in case it does we print it, but doesn't really matter
                    print(f"Note: Stimulation number does not match the expected number counted {counted_stim_number} != found {stim_number}")
                sham_stims[counted_stim_number]["begin"] = annot["onset"]
            elif match.group("type") in [ "end" ]:
                sham_stims[counted_stim_number]["end"] = annot["onset"]
                counted_stim_number += 1
            else:
                # should not happen
                print("Error: unknown type of stimulation annotation")

    durations = []
    for stim_number, stim in sham_stims.items():
        
        # if both begin and end are found, we can just use these
        if "begin" in stim and "end" in stim:
            begin = stim["begin"] - pre_stim_buffer
            end = stim["end"] + post_stim_buffer
        elif "begin" in stim: # if only begin is found, we use the sham_stim_duration to determine the end
            begin = stim["begin"] - pre_stim_buffer
            end = begin + sham_stim_duration + post_stim_buffer
        else: # TODO: this is a quick fix, might not be perfect..
            begin = 0
            end = 0
            print("Error: Sham stimulation without begin or end found")
        
        duration = end - begin
        durations.append(duration)
        stim_annots.append(begin, duration, f"BAD_Active_Stimulation {stim_number}")
        stim_annots.append(end, analy_duration, f"Post_Stimulation {stim_number}")
        if not post_only or stim_number == 1:
            stim_annots.append(begin - analy_duration, analy_duration, f"Pre_Stimulation {stim_number}")

    print(durations)
    print(f"Standard deviation (of durations):{np.std(durations)}")
    print(f"Found {len(durations)} sham stimulation intervals.")

     # last post-stimulation period should handled differently
    last_annot = stim_annots[-1].copy()

    # instead we decide to just add a new annotation that extends to the end of the recording
    last_analy_duration = raw._last_time - (last_annot["onset"] + analy_duration)
    stim_annots.append(last_annot["onset"] + analy_duration, last_analy_duration, "Post_Stimulation_ToEnd")

    raw.set_annotations(stim_annots)
    stim_annots.save(join(proc_dir, outfilename), overwrite=overwrite)

    if testing:
        # plot (optional)
        raw.pick(picks).plot(block=True, scalings=dict(eeg=20e-3), duration=700)


# In[ ]:


number_of_preprocessed_files = 0

for subdir in sub_dirs:
    if subdir in skipDir:
        continue
    print(f"{subdir}".center(80, '-'))
    proclist = listdir(join(proc_dir, subdir)) # and in proc directory

    print(number_of_preprocessed_files)

    for file in proclist:
        # if we are testing, only apply preprocessing to one file
        if testing and number_of_preprocessed_files > 0:
            continue

        match = re.match("NAP_(?P<subj>\\d{4})_(?P<cond>T\\d{1})-rp.fif", file)
        if not match:
            continue
        (subj, cond) = match.groups()
        
        if [subj, cond] in exclude:
            continue
        if include and [subj, cond] not in include:
            continue
        
        outname = f"NAP_{subj}_{cond}-rpa.fif"
        if outname in proclist and not overwrite:
            print(f"{outname} already exists. Skipping...")
            continue
           
        # now do the actual processing
        # ----------------------------
        excludeCH = []
        if "_".join([subj,cond]) in ses_bad_ch:
            excludeCH = ses_bad_ch.get("_".join([subj,cond]))
            

        if subdir == 'sotDCS_anod' or subdir == 'sotDCS_cat':
            detect_sotDCS_stimulation(join(proc_dir, subdir, file), join(proc_dir, subdir, outname),excludeCH)
        elif subdir == 'tDCS': 
            detect_tDCS_stimulation(join(proc_dir, subdir, file), join(proc_dir, subdir, outname))
        elif subdir == 'sham': 
            detect_sham_stimulation(join(proc_dir, subdir, file), join(proc_dir, subdir, outname))

        number_of_preprocessed_files += 1


# In[ ]:




