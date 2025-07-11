#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
from mne.stats.cluster_level import _find_clusters
from joblib import Parallel, delayed
from mne.stats import fdr_correction
from tensorpac import EventRelatedPac as ERPAC
from mne.time_frequency import tfr_morlet
#from tensorpac import PreferredPhase as PP
import pandas as pd
import numpy as np
from scipy.stats import norm
from os.path import join
import matplotlib.pyplot as plt
import pickle
plt.ion()
import matplotlib
matplotlib.use('agg')
font = {'weight' : 'bold',
        'size'   : 48}
matplotlib.rc('font', **font)


# In[ ]:


def do_erpac(ep, epo, cut, baseline=None, fit_args={"mcp":"fdr", "p":0.05,
                                                    "n_jobs":1,
                                                    "method":"circular",
                                                    "n_perm":1000}):
    data = epo.get_data()[:,0,] * 1e+6
    phase = ep.filter(epo.info["sfreq"], data, ftype="phase", n_jobs=n_jobs)
    power = ep.filter(epo.info["sfreq"], data, ftype="amplitude", n_jobs=n_jobs)
    if baseline:
        base_inds = epo.time_as_index((baseline[0], baseline[1]))
        bl = power[...,base_inds[0]:base_inds[1]]
        bl_mu = bl.mean(axis=-1, keepdims=True)
        bl_std = bl.std(axis=-1, keepdims=True)
        power = (power - bl_mu) / bl_std
    cut_inds = epo.time_as_index((cut[0], cut[1]))
    power = power[...,cut_inds[0]:cut_inds[1]]
    phase = phase[...,cut_inds[0]:cut_inds[1]]
    erpac = ep.fit(phase, power, **fit_args)
    times = epo.times[cut_inds[0]:cut_inds[1]]
    n = phase.shape[1]
    return erpac, times, n


# In[ ]:


def compare_rho(erpac_a, n_a, erpac_b, n_b, fdr=0.05):
    erpac_a_fish = np.arctanh(erpac_a)
    erpac_b_fish = np.arctanh(erpac_b)
    erpac_delt = erpac_b_fish - erpac_a_fish
    delt_se = np.sqrt(1/(n_a-3) + 1/(n_b-3))
    erpac_z = erpac_delt / delt_se
    erpac_p = norm.sf(abs(erpac_z))*2
    if fdr:
        erpac_p = fdr_correction(erpac_p, alpha=fdr)[1]
    return erpac_z, erpac_p


# In[ ]:


def tfce_correct(data, tfce_thresh=None):
    if tfce_thresh is None:
        tfce_thresh = dict(start=0, step=0.2)
    pos_data = data.copy()
    pos_data[pos_data<0] = 0
    neg_data = data.copy()
    neg_data[neg_data>0] = 0
    pos_clusts = _find_clusters(pos_data, tfce_thresh)[1].reshape(data.shape)
    neg_clusts = _find_clusters(neg_data, tfce_thresh)[1].reshape(data.shape)
    out_data = np.zeros_like(data) + pos_clusts - neg_clusts
    return out_data


# In[ ]:


def permute(perm_idx, a_n, epo_inds, phase, power, fit_args, subj_inds=None):
    print("Permutation {} of {}".format(perm_idx, n_perm))
    # if we have group as a factor, we shuffle data only within subjects
    if subj_inds is not None:
        subjs = list(np.unique(subj_inds))
        for subj in subjs:
            subj_inds = subj_inds==subj
            these_epo_inds = epo_inds[subj_inds].copy()
            np.random.shuffle(these_epo_inds)
            epo_inds[subj_inds] = these_epo_inds
        else:
            np.random.shuffle(epo_inds)
    erpac_a = ep.fit(phase[:,epo_inds[:a_n],], power[:,epo_inds[:a_n],], **fit_args)
    erpac_b = ep.fit(phase[:,epo_inds[a_n:],], power[:,epo_inds[a_n:],], **fit_args)
    erpac_z, _ = compare_rho(erpac_a, a_n, erpac_b, len(epo_inds)-a_n,
                             fdr=None)
    erpac_c = tfce_correct(erpac_z)
    return abs(erpac_c).max()


# In[ ]:


def do_erpac_perm(epo_a, epo_b, cut, baseline=None, n_perm=1000, n_jobs=1,
                  fit_args={"mcp":"fdr", "p":0.05, "n_jobs":1,
                            "method":"circular"}):
    data_a = epo_a.get_data()[:,0,] * 1e+6
    data_b = epo_b.get_data()[:,0,] * 1e+6
    data = np.vstack((data_a, data_b))
    epo_inds = np.arange(len(data))
    perm_maxima, perm_minima = np.zeros(n_perm), np.zeros(n_perm)
    a_n = len(epo_a)
    subj_inds = np.hstack((epo_a.metadata["Subj"].values,
                           epo_b.metadata["Subj"].values))
    phase = ep.filter(cond_epo.info["sfreq"], data, ftype="phase", n_jobs=n_jobs)
    power = ep.filter(cond_epo.info["sfreq"], data, ftype="amplitude", n_jobs=n_jobs)
    if baseline:
        base_inds = cond_epo.time_as_index((baseline[0], baseline[1]))
        bl = power[...,base_inds[0]:base_inds[1]]
        bl_mu = bl.mean(axis=-1, keepdims=True)
        bl_std = bl.std(axis=-1, keepdims=True)
        power = (power - bl_mu) / bl_std
    cut_inds = epo.time_as_index((cut[0], cut[1]))
    power = power[...,cut_inds[0]:cut_inds[1]]
    phase = phase[...,cut_inds[0]:cut_inds[1]]
    times = epo.times[cut_inds[0]:cut_inds[1]]
    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(permute)(
                       i, a_n, epo_inds, phase, power, fit_args, subj_inds)
                       for i in range(n_perm))
    return results


# In[ ]:


if __name__ == "__main__":
    root_dir = "/media/Linux6_Data/DATA/SFB2"
    proc_dir = join(root_dir,"proc")
    fig_dir = join(root_dir, "proc","figs")
    phase_freqs = {"SO":(0.5, 1.25),"deltO":(1.25, 4)}
    power_freqs = (5, 25)
    osc_cuts = {"SO":(-1.5,1.5),"deltO":(-1,1)}
    baseline = (-2.35, -1.5)
    method = "wavelet"
    n_jobs = 24
    p = 0.05
    n_perm = 2048
    
    sfreq = 100
    tfce_thresh = dict(start=0, step=0.2)
    f_amp = np.linspace(power_freqs[0], power_freqs[1], 50)
    epo = mne.read_epochs(join(proc_dir, "grand-epo.fif"), preload=True)
    
    epo.resample(sfreq, n_jobs=n_jobs)
    epos = []
    dfs = []
    osc = "SO"
    recalc = True
    for polarity in ["cathodal", "anodal"]:
        for ROI in ["frontal", "parietal"]:
            print(len(epo[f"OscType=='{osc}' and Polarity=='{polarity}' and ROI=='{ROI}' and Index!=99"]))
            if len(epo[f"OscType=='{osc}' and Polarity=='{polarity}' and ROI=='{ROI}' and Index!=99"]):
                this_epo = epo.copy()[f"OscType=='{osc}' and Polarity=='{polarity}' and ROI=='{ROI}' and Index!=99"]
                this_epo.pick_channels([ROI])
                pf = phase_freqs[osc]
                osc_cut = osc_cuts[osc]
                ep = ERPAC(f_pha=pf, f_amp=f_amp, dcomplex=method)
                sham_epo = this_epo.copy()["Cond=='sham'"]
                sham_erpac, times, sham_n = do_erpac(ep, sham_epo, osc_cut, baseline=baseline)
                erpacs = []
                ns = []
                cond_epo = this_epo.copy()["Cond=='SOstim'"]
                erpac, times, n = do_erpac(ep, cond_epo, osc_cut, baseline=baseline)
                erpacs.append(erpac)
                ns.append(n)
                erpac_z, erpac_p = compare_rho(sham_erpac, sham_n, erpac, n, fdr=None)
                erpac_z = erpac_z.squeeze()
                erpac_c = _find_clusters(erpac_z, threshold=tfce_thresh)
                erpac_c = np.reshape(erpac_c[1], erpac_z.shape)

                #ep.pacplot(erpac_c, times, ep.yvec)
                if recalc:
                    results = do_erpac_perm(sham_epo, cond_epo, osc_cut, baseline=baseline,
                                            n_perm=n_perm, n_jobs=n_jobs)
                    results = np.array(results)
                    np.save(join(proc_dir, f"erpac_perm_{ROI}_{polarity}_onlyPost.npy"), results)
                else:
                    results = np.load(join(proc_dir, f"erpac_perm_{ROI}_{polarity}_onlyPost.npy"))
                thresh_val = np.quantile(results, 1-p/2)
                erpac_mask = abs(erpac_c) > thresh_val

                # make mne tfr template for plotting
                e = this_epo[0].crop(tmin=osc_cut[0], tmax=osc_cut[1]-1/sfreq)
                tfr = tfr_morlet(e, f_amp[:-1], n_cycles=5, average=False, return_itc=False)
                tfr = tfr.average()
                fig, ax = plt.subplots(figsize=(19.2,19.2))
                tfr.data[0,:,:] = erpac_z.squeeze()
                tfr.plot(mask=erpac_mask, mask_style="contour", cmap="inferno",
                        vmin=-3, vmax=3, axes=ax, picks=ROI)
                plt.ylabel("Frequency (Hz)")
                plt.xlabel("Time (s)")
                ax.set_xticks([-1, 0, 1])
                ax.set_xticklabels([-1, 0, 1], fontweight="normal")
                ax.set_yticks([10, 15, 20])
                ax.set_yticklabels([10, 15, 20], fontweight="normal")
                cbar = ax.images[-1].colorbar
                fig.axes[-1].set_ylabel("Normalised difference")
                cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
                fig.axes[-1].set_yticklabels([-3, -2, -1, 0, 1, 2, 3], fontweight="normal")
                plt.ylim(8, 22)
                cut_inds = epo.time_as_index((osc_cut[0], osc_cut[1]))
                evo = cond_epo.average().data[0,cut_inds[0]:cut_inds[1]]
                evo = (evo - evo.min())/(evo.max()-evo.min())
                evo = evo*5 + 11
                plt.plot(times, evo, linewidth=10, color="gray", alpha=0.8)
                plt.suptitle(f"SO ERPAC, {ROI}, {polarity} stimulation\nnormalised difference: Stim - Sham", fontsize=40)
                plt.savefig(join(fig_dir, f"ERPAC_sfb2_{ROI}_{polarity}_onlyPost.png"))
                plt.savefig(join(fig_dir, f"ERPAC_sfb2_{ROI}_{polarity}_onlyPost.svg"))
    print("done")

