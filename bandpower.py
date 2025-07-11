import mne 
import numpy as np
from os.path import join
from os import listdir
import re
import pandas as pd
from scipy.integrate import simps

def calc_bandpower(inst, picks, bands, n_fft=256, n_per_seg=None,
                   n_jobs=1, log=False, relative=False, average="mean"):
    if isinstance(inst, mne.io.BaseRaw):
        epo = mne.make_fixed_length_epochs(inst, duration=30)
    else:
        epo = inst
    output = {"chan_names":picks}
    min_freq = np.array([x[0] for x in bands.values()]).min()
    max_freq = np.array([x[1] for x in bands.values()]).max()
    psd = epo.compute_psd(fmax=max_freq, method="welch", n_fft=n_fft, picks=picks)
    (psd, freqs) = psd.get_data().mean(axis=1), psd.freqs
    psd *= 1e12
    freq_res = freqs[1] - freqs[0]
    for band_k, band_v in bands.items():
        fmin, fmax = band_v
        inds = np.where((freqs>=fmin) & (freqs<=fmax))[0]
        power = simps(psd[...,inds], dx=freq_res)
        if relative:
            full_power = simps(psd, dx=freq_res)
            power = power / full_power
        if log and not relative:
            power = np.log10(power)
        if average == "mean":
            output[band_k] = power.mean(axis=0)
        elif average == "median":
            output[band_k] = power.median(axis=0)
    return output

chan_groups = {"frontal":["Fz", "FC1", "FC2"],
               "parietal":["Cz", "CP1", "CP2"]}
bands = {"SO":[0.5, 1.25], "deltO":[1, 4], "12-15":[12, 15], "15-18":[15, 18]}

root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")
n_jobs = 12
n_fft = 4096

df_dict = {"Subject":[], "Condition":[], "ROI":[]}
for band in bands.keys():
    df_dict[band] = []
    df_dict[f"{band}_rel"] = []

filelist = listdir(proc_dir)
for filename in filelist:
    this_match = re.match("cp_NAP_(\d{4})_(.*)-raw.fif", filename)
    if not this_match:
        continue
    (subj, cond) = this_match.groups()
    c_groups = chan_groups.copy()

    # special case
    if subj == "1021":
        if cond == "sham" or cond == "anodal":
            c_groups["parietal"] = ["P1", "P2", "CPz"]
            c_groups["frontal"] = ["F1", "Iz", "F2"]

    raw = mne.io.Raw(join(proc_dir, filename), preload=True)

    for k,v in c_groups.items():
        output = calc_bandpower(raw, v, bands, n_fft=n_fft, n_jobs=n_jobs)
        del output["chan_names"]
        output_rel = calc_bandpower(raw, v, bands, n_fft=n_fft, relative=True, n_jobs=n_jobs)
        del output_rel["chan_names"]
        df_dict["Subject"].append(subj)
        df_dict["Condition"].append(cond)
        df_dict["ROI"].append(k)
        for kk, vv in output.items():
            df_dict[kk].append(vv)
        for kk, vv in output_rel.items():
            df_dict[f"{kk}_rel"].append(vv)

df = pd.DataFrame.from_dict(df_dict)
df = df.pivot_table(index="Subject", columns=["ROI", "Condition"], values=list(bands.keys()))
df = df.sort_values(["Subject"])
df.to_csv(join(proc_dir, "bandpower.csv"))