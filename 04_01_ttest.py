#!/usr/bin/env python
# coding: utf-8

# ## Summary of Key Steps:
# 1) Paired t-tests compare stimulation conditions.
# 2) Permutation tests (if enabled) generate a null distribution for statistical significance.
# 3) Results are saved or loaded to avoid redundant computations.
# 4) Sample sizes are computed for proper labeling.
# 5) Grand averages are computed across trials/participants.
# 6) Time-frequency representations (TFRs) are plotted for each condition.
# 7) Significant clusters are identified based on permutation-derived thresholds.
# 8) A significance mask is applied to highlight important differences.
# 9) A difference plot is generated, showing significant time-frequency regions.

# In[ ]:


from mne.time_frequency import tfr_morlet
import numpy as np
import mne
from os.path import join
from scipy.stats import ttest_rel
from erpac import tfce_correct
import matplotlib.pyplot as plt
from mne.stats.cluster_level import _find_clusters
from joblib import Parallel, delayed


# In[ ]:


def norm_overlay(x, min=10, max=20, centre=15, xmax=4e-05):
    x = x / xmax
    x = x * (max-min)/2 + centre
    return x


# In[ ]:


#def permute(perm_idx, a, b, perm_n):
def permute(perm_idx, a, b):
    print("Permutation {} of {}".format(perm_idx, perm_n))
    swap_inds = np.random.randint(0,2, size=len(a)).astype(bool)
    swap_a, swap_b  = a.copy(), b.copy()
    swap_a[swap_inds,] = b[swap_inds,]
    swap_b[swap_inds,] = a[swap_inds,]
    t = ttest_rel(swap_a, swap_b, axis=0)
    try:
        clusts = tfce_correct(t[0][0,])
    except:
        breakpoint()
    return abs(clusts).max()


# In[ ]:


root_dir = "/media/Linux6_Data/DATA/SFB2"
proc_dir = join(root_dir, "proc")
fig_dir = join(root_dir, "proc", "figs")
min_freq, max_freq = 12, 18 
freqs = np.linspace(min_freq, max_freq, 50)
n_cycles = 5
n_jobs = 24
threshold = 0.05
#threshold = 0.98
vmin, vmax = -3, 3
tfce_thresh = dict(start=0, step=0.2)
#tfce_thresh = dict(start=0.5, step=0.05)
analy_crop = [-1, 1]
#print(tfce_thresh)


# In[ ]:


do_permute = True
perm_n = 1024
#perm_n = 4096


# In[ ]:


ur_epo = mne.read_epochs(join(proc_dir, f"grand-epo.fif"))
ur_epo = ur_epo["OscType=='SO'"]
subjs = list(ur_epo.metadata["Subj"].unique())
ROIs = list(ur_epo.metadata["ROI"].unique())


# In[ ]:


for ROI in ROIs:
    epo = ur_epo.copy()[f"ROI=='{ROI}'"]
    epo.pick_channels([ROI])
    tfr = tfr_morlet(epo, freqs, n_cycles, return_itc=False, average=False, output="power",
                    n_jobs=n_jobs)
    tfr.apply_baseline((-2.25, -1), mode="zscore")
    tfr.crop(*analy_crop)
    epo.crop(*analy_crop)
    subjs = list(ur_epo.metadata["Subj"].unique())
    so_stim_an, so_stim_ca, so_sham_an, so_sham_ca = [], [], [], []
    tfr_stim_an, tfr_stim_ca, tfr_sham_an, tfr_sham_ca = [], [], [], []
    for subj in subjs:
        subj_epo = epo.copy()[f"Subj=='{subj}'"]
        tfr_epo = tfr.copy()[f"Subj=='{subj}'"]
        if 0 in (
            len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal' and Index!=99"]),  
            len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal' and Index!=99"]),
            len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal' and Index!=99"])
            ):
            print("crap")
            continue

        # get SO for overlays
        so_stim_an.append(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal' and Index!=99"].average())
        so_stim_ca.append(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal' and Index!=99"].average())
        so_sham_an.append(subj_epo.copy()["Cond=='sham' and Polarity=='anodal' and Index!=99"].average())
        tfr_stim_an.append(tfr_epo.copy()["Cond=='SOstim' and Polarity=='anodal' and Index!=99"].average())
        tfr_stim_ca.append(tfr_epo.copy()["Cond=='SOstim' and Polarity=='cathodal' and Index!=99"].average())
        tfr_sham_an.append(tfr_epo.copy()["Cond=='sham' and Polarity=='anodal' and Index!=99"].average())
    so_stim_an = norm_overlay(mne.grand_average(so_stim_an).data.squeeze(),
                              min=min_freq, max=max_freq)
    so_stim_ca = norm_overlay(mne.grand_average(so_stim_ca).data.squeeze(),
                              min=min_freq, max=max_freq)
    so_sham_an = norm_overlay(mne.grand_average(so_sham_an).data.squeeze(),
                              min=min_freq, max=max_freq)
    
    tfr_stim_an_dat = np.array([t.data for t in tfr_stim_an])
    tfr_stim_ca_dat = np.array([t.data for t in tfr_stim_ca])
    tfr_sham_an_dat = np.array([t.data for t in tfr_sham_an])

    # 1. Perform paired t-tests to compare conditions
    an_t = ttest_rel(tfr_stim_an_dat, tfr_sham_an_dat, axis=0) # Compare "Stimulation Anodal" vs. "Sham Anodal"
    ca_t = ttest_rel(tfr_stim_ca_dat, tfr_sham_an_dat, axis=0) # Compare "Stimulation Cathodal" vs. "Sham Anodal"

    # 2. Conduct permutation tests (optional, if do_permute is True)
    if do_permute:
        results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(permute)( # Parallel computation to speed up the process
                             i, tfr_stim_an_dat, tfr_sham_an_dat)       # Permute data labels and compute t-test
                             for i in range(perm_n))                    # Repeat permutation test perm_n times (e.g., 1000 or 5000 iterations)
        #results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(permute)( # Parallel computation to speed up the process
        #                     i, tfr_stim_an_dat, tfr_sham_an_dat,perm_n)       # Permute data labels and compute t-test
        #                     for i in range(perm_n))                    # Repeat permutation test perm_n times (e.g., 1000 or 5000 iterations)
        an_results = np.array(results)                                  # Store permutation results as a NumPy array
        np.save(join(proc_dir, f"erpac_perm_{ROI}_an.npy"), results)    # Save results for future use
        # Repeat permutation test for the Cathodal stimulation condition
        results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(permute)(
                             i, tfr_stim_ca_dat, tfr_sham_an_dat)
                             for i in range(perm_n))
        ca_results = np.array(results)                                  # Store permutation results for Cathodal condition
        np.save(join(proc_dir, f"erpac_perm_{ROI}_ca.npy"), results)    # Save Cathodal permutation results
    # 3. Load previously saved permutation results if they already exist (to save computation time)
    else:
        an_results = np.load(join(proc_dir, f"erpac_perm_{ROI}_an.npy"))    # Load Anodal permutation results
        ca_results = np.load(join(proc_dir, f"erpac_perm_{ROI}_ca.npy"))    # Load Cathodal permutation results
    # 4. Compute the number of samples (participants/trials) in each condition
    stim_an_n = len(tfr_stim_an)    # Number of samples in Stimulation Anodal condition
    stim_ca_n = len(tfr_stim_ca)    # Number of samples in Stimulation Cathodal condition
    sham_an_n = len(tfr_sham_an)    # Number of samples in Sham Anodal condition
    sham_ca_n = len(tfr_sham_ca)    # Number of samples in Sham Cathodal condition
    # 5. Compute the grand average across trials/participants for each condition
    tfr_stim_an = mne.grand_average(tfr_stim_an)    # Compute the grand average for "Stimulation Anodal"
    tfr_stim_ca = mne.grand_average(tfr_stim_ca)    # Compute the grand average for "Stimulation Cathodal"
    tfr_sham_an = mne.grand_average(tfr_sham_an)    # Compute the grand average for "Sham Anodal"
    # 6. Create plots for visualization
    fig, axes = plt.subplots(2, 3, figsize=(38.4, 21.6))                # Create a figure with 2 rows and 3 columns
         # Plot "Stimulation Anodal" time-frequency representation (TFR)
    tfr_stim_an.plot(vmin=0, vmax=vmax, axes=axes[0,0], cmap="hot")     # Use a heatmap color scheme
    axes[0,0].set_title(f"Stimulation Anodal ({stim_an_n})")            # Add title with sample size
    axes[0,0].plot(tfr_stim_an.times, so_stim_an, color="white")        # Overlay time series data
        # Plot "Sham Anodal" TFR
    tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes[0,1], cmap="hot")     
    axes[0,1].set_title(f"Sham Anodal ({sham_an_n})")
    axes[0,1].plot(tfr_sham_an.times, so_sham_an, color="white")
    # 7. Identify significant clusters using thresholding
    an_c = _find_clusters(an_t[0][0,], threshold=tfce_thresh)   # Find clusters in t-values
    an_c = np.reshape(an_c[1], an_t[0].shape)                   # Reshape the result to match the original data
    # Compute threshold value from permutation results
    thresh_val = np.quantile(an_results, 1-threshold/2)         # Determine significance threshold from permutation distribution
    # 8. Create a mask for significant areas in the data
    mask = abs(an_c) > thresh_val       # Mask areas where absolute cluster values exceed threshold
    mask_2d = mask[0, :, :]             # Convert to 2D for plotting purposes
    # 9. Plot the difference between "Stimulation Anodal" and "Sham Anodal" with significance masking
    (tfr_stim_an-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes[0,2], mask=mask_2d)
    #(tfr_stim_an-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes[0,2],
    #                              mask_style="contour", mask=mask)
    axes[0,2].set_title("Stim Anodal - Sham Anodal")
    axes[0,2].plot(tfr_stim_an.times, so_stim_an, color="black")
    tfr_stim_ca.plot(vmin=0, vmax=vmax, axes=axes[1,0], cmap="hot")
    axes[1,0].set_title(f"Stimulation Cathodal ({stim_an_n})")
    axes[1,0].plot(tfr_stim_ca.times, so_stim_ca, color="white")
    tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes[1,1], cmap="hot")
    axes[1,1].set_title(f"Sham Anodal ({sham_an_n})")
    axes[1,1].plot(tfr_sham_an.times, so_sham_an, color="white")
    ca_c = _find_clusters(ca_t[0][0,], threshold=tfce_thresh)
    ca_c = np.reshape(ca_c[1], ca_t[0].shape)
    thresh_val = np.quantile(ca_results, 1-threshold/2)
    mask = abs(ca_c) > thresh_val
    mask_2d = mask[0, :, :]
    (tfr_stim_ca-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes[1,2], mask=mask_2d)
    #(tfr_stim_ca-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes[1,2],
    #                                mask_style="contour", mask=mask)
    axes[1,2].set_title("Stim Cathodal - Sham Anodal")
    axes[1,2].plot(tfr_stim_an.times, so_stim_ca, color="black")
    plt.suptitle(ROI)
    plt.tight_layout()
    plt.savefig(join(fig_dir, f"ttest_{ROI}.png"))


# In[ ]:


print(mask.shape)
print(tfr_stim_ca.shape)
print((tfr_stim_an-tfr_sham_an).shape)


# In[ ]:





# In[ ]:




