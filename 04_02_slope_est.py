import mne 
import numpy as np
from os.path import join
from os import listdir
import pandas as pd
import re
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")
fig_dir = join(root_dir, "figs")

chan_groups = {"frontal":["Fz", "FC1","FC2"],
               "parietal":["Cz","CP1","CP2"]}

calc = True

if calc:
    filelist = listdir(proc_dir)
    df_dict = {"Subj":[], "Cond":[], "ROI":[], "LogPower":[], "Freqs":[],}
    for filename in filelist:
        this_match = re.match("cp_NAP_(\d{4})_(.*)-raw.fif", filename)
        if not this_match:
            continue
        (subj, cond) = this_match.groups()
        if cond == "sfb1":
            continue

        c_groups = chan_groups.copy()
        # special case
        if subj == "1021":
            if cond == "sham" or cond == "anodal":
                c_groups["parietal"] = ["P1", "P2", "CPz"]
                c_groups["frontal"] = ["F1", "Iz", "F2"]

        raw = mne.io.Raw(join(proc_dir, filename), preload=True)
        picks = mne.pick_types(raw.info, eeg=True)

        passed = np.zeros(len(c_groups), dtype=bool)
        for idx, (k,v) in enumerate(c_groups.items()):
            pick_list = [vv for vv in v if vv not in raw.info["bads"]]
            if not len(pick_list):
                print("No valid channels")
                continue
            avg_signal = raw.get_data(pick_list).mean(axis=0, keepdims=True)
            avg_info = mne.create_info([k], raw.info["sfreq"], ch_types="eeg")
            avg_raw = mne.io.RawArray(avg_signal, avg_info)
            raw.add_channels([avg_raw], force_update_info=True)
            passed[idx] = 1
        if not all(passed):
            print("Could not produce valid ROIs")
            continue
        # ROIs only, drop everything els
        raw.pick_channels(list(chan_groups.keys()))

        # sort out conditiona and polarity
        if cond == "anodal":
            cond = "stim anodal"
        elif cond == "cathodal":
            cond = "stim cathodal"
        elif cond == "sham":
            pass
        else:
            raise ValueError("Could not organise condition/polarity")

        for ROI in ["frontal", "parietal"]:
            this_raw = raw.copy().pick_channels([ROI])
            psd = this_raw.compute_psd(fmin=0.5, fmax=45, method="welch", n_fft=4096)
            df_dict["Freqs"].extend(psd.freqs)
            logpow = np.log10(psd.get_data()[0])
            df_dict["LogPower"].extend(logpow)
            df_dict["Subj"].extend([subj]*len(psd.freqs))
            df_dict["Cond"].extend([cond]*len(psd.freqs))
            df_dict["ROI"].extend([ROI]*len(psd.freqs))
        
    df = pd.DataFrame.from_dict(df_dict)
    df.to_pickle(join(proc_dir, "slope_df.pickle"))

df = pd.read_pickle(join(proc_dir, "slope_df.pickle"))
for ROI in ["frontal", "parietal"]:
    plt.figure()
    this_df = df.query(f"ROI=='{ROI}'")
    rlm_model = smf.rlm("LogPower ~ Freqs*C(Cond, Treatment(reference='sham')) + Subj", this_df)
    results = rlm_model.fit()
    print(ROI)
    print(results.summary())
    sns.lineplot(x="Freqs", y="LogPower", hue="Cond", data=this_df)
    plt.title(ROI)
    plt.savefig(join(fig_dir, f"log_power_{ROI}.png"))
    

