import mne
from os import listdir
from mne.stats.cluster_level import _find_clusters
import re
from os.path import join
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import pickle
plt.ion()

def get_stat_array(modfit, exog_names, dims):
    # converts the output of lmm_mass_univ.py into arrays of parameters, t-values, and p-values,
    # arranged by conditions into dictionaries
    params = {ex_n:[] for ex_n in exog_names}
    tvals = {ex_n:[] for ex_n in exog_names}
    pvals = {ex_n:[] for ex_n in exog_names}
    for mf in modfit:
        for ex_n in exog_names:
            params[ex_n].append(mf.params[exog_names.index(ex_n)])
            tvals[ex_n].append(mf.tvalues[exog_names.index(ex_n)])
            pvals[ex_n].append(mf.pvalues[exog_names.index(ex_n)])
    # convert to array, reshape
    params = {k:np.array(v).reshape(dims) for k,v in params.items()}
    tvals = {k:np.array(v).reshape(dims) for k,v in tvals.items()}
    pvals = {k:np.array(v).reshape(dims) for k,v in pvals.items()}

    return params, tvals, pvals
    
def norm_overlay(x, min=10, max=20, centre=15, xmax=4e-05):
    x = x / xmax
    x = x * (max-min)/2 + centre
    return x

root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")
fig_dir = join(root_dir, "figs")

freqs = np.linspace(10, 20, 25)
n_cycles = 5
n_jobs = 24
tfce_thresh = dict(start=0, step=0.2)

test_key = {
    "Stim":"C(Cond, Treatment('sham'))[T.stim]",
    "Polarity":'Polarity[T.cathodal]',
    "Stim*Polarity":"C(Cond, Treatment('sham'))[T.stim]:Polarity[T.cathodal]",
    "Gap":"Gap",
    "Gap*Stim":"C(Cond, Treatment('sham'))[T.stim]:Gap",
    "Gap*Polarity:":'Polarity[T.cathodal]:Gap',
    "Gap*Polarity*Stim":"C(Cond, Treatment('sham'))[T.stim]:Polarity[T.cathodal]:Gap"
}

ur_epo = mne.read_epochs(join(proc_dir, f"grand-epo.fif"))
ROIs = list(ur_epo.metadata["ROI"].unique())
ur_epo = ur_epo["OscType=='SO'"]
#ROIs = ["frontal"]
for ROI in ROIs:
    epo = ur_epo.copy()
    epo = epo[f"ROI=='{ROI}'"]
    epo.pick_channels([ROI])
    tfr = tfr_morlet(epo, freqs, n_cycles, return_itc=False, average=False, output="power",
                    n_jobs=n_jobs)
    tfr.crop(-2.25, 2.25)
    tfr.apply_baseline((-2.25, -1), mode="zscore")
    tfr = tfr.decimate(2)
    epo = epo.decimate(2)

    # get stat results  (produced by lmm_mass_univ.py)
    with open(join(proc_dir, f"lmm_fits_{ROI}.pickle"), "rb") as f:
        fits = pickle.load(f)
    exog_names = fits["exog_names"]
    modfit = fits["fits"]
    dims = fits["data_dims"]
    params, tvals, pvals = get_stat_array(modfit, exog_names, dims)

    # get SO for overlays
    epo.crop(-2.25, 2.25)
    so_overlay = norm_overlay(epo.average().data.squeeze())
    line_color = "green"

    # define layout of graph
    fig, axes = plt.subplots(2, 4, figsize=(38.4, 21.6))
    it_axes = iter(axes.flatten())
    vmin, vmax = -3, 3
    tfr = tfr.average()

    for k,v in test_key.items():
        ax = next(it_axes)
        tval = tvals[v]
        tfr.data = tval.T[None,]
        p_mask = pvals[v]<0.05
        tfr.plot(vmin=vmin, vmax=vmax, axes=ax, cmap="seismic", mask=p_mask.T, mask_style="contour")
        ax.plot(tfr.times, so_overlay, color=line_color, linewidth=6)
        #ax.set_facecolor("black")
        ax.set_title(k)
    next(it_axes).axis("off")

    plt.suptitle(ROI)
    plt.savefig(join(fig_dir, f"lme_{ROI}.png"))