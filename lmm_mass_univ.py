import mne
from mne.time_frequency import tfr_morlet
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import argparse
import pandas as pd
from os.path import join
import re
import numpy as np
import matplotlib.pyplot as plt
from mne.stats.cluster_level import _find_clusters
import pickle
plt.ion()
import warnings
warnings.filterwarnings("ignore")

def mass_uv_lmm(data, endog, exog, groups, exog_re, exog_vc):
    exog_n = exog.shape[1]
    dat = data.reshape(len(data), data.shape[1]*data.shape[2])
    fits = []
    for pnt_idx in range(dat.shape[1]):
        print("{} of {}".format(pnt_idx, dat.shape[1]))
        endog = dat[:, pnt_idx]
        this_mod = MixedLM(endog, exog, groups, exog_re=exog_re, exog_vc=exog_vc)
        try:
            fits.append(this_mod.fit(reml=True))
        except:
            print("\nCoudn't fit model.\n")
            fits.append(None)
    return fits

root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")

freqs = np.linspace(10, 20, 25)
n_cycles = 5
n_jobs = 24

ur_epo = mne.read_epochs(join(proc_dir, f"grand-epo.fif"))
ROIs = list(ur_epo.metadata["ROI"].unique())
ur_epo = ur_epo["OscType=='SO'"]
for ROI in ROIs:
    epo = ur_epo.copy()
    epo = epo[f"ROI=='{ROI}'"]
    epo.pick_channels([ROI])
    tfr = tfr_morlet(epo, freqs, n_cycles, return_itc=False, average=False, output="power",
                    n_jobs=n_jobs)
    tfr.crop(-2.25, 2.25)
    tfr.apply_baseline((-2.25, -1), mode="zscore")
    tfr = tfr.decimate(2)

    subjs = np.unique(tfr.metadata["Subj"].values)
    df = tfr.metadata.copy()
    df["Brain"] = np.zeros(len(df),dtype=np.float64)
    data = np.swapaxes(tfr.data[:,0],1,2)

    formula = "Brain ~ C(Cond, Treatment('sham'))*Polarity*Gap"
    md = smf.mixedlm(formula, df, groups=df["Subj"])
    (endog, exog, groups, exog_names, exog_re, exog_vc) = (
                                                           md.endog,
                                                           md.exog,
                                                           md.groups,
                                                           md.exog_names,
                                                           md.exog_re,
                                                           md.exog_vc
                                                           )

    fits = mass_uv_lmm(data, endog, exog, groups, exog_re, exog_vc)
    fits = {"exog_names":exog_names, "fits":fits, 
            "data_dims":data.shape[1:]}
    outpath = join(proc_dir, f"lmm_fits_{ROI}.pickle")
    with open(outpath, "wb") as f:
                pickle.dump(fits, f)
    del fits
