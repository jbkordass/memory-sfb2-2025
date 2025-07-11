#!/usr/bin/env python
# coding: utf-8

# In[201]:


import mne
import numpy as np
import pandas as pd
from scipy.stats import circmean
from os.path import join
import statsmodels.formula.api as smf
import seaborn as sns
from circular_hist import circ_hist_norm
import matplotlib.pyplot as plt

#from pycircstat.tests import watson_williams, vtest, kuiper
#from astropy.stats import kuiper_two

plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 28}
matplotlib.rc('font', **font)


# In[202]:


def r_vector(rad):
    x_bar, y_bar = np.cos(rad).mean(), np.sin(rad).mean()
    r_mean = circmean(rad, low=-np.pi, high=np.pi)
    r_norm = np.linalg.norm((x_bar, y_bar))
    return r_mean, r_norm


# In[203]:


def r_vector_list(radians):
    r_mean = []
    r_norm = []
    for rad in radians:
        x_bar, y_bar = np.cos(rad), np.sin(rad)
        r_mean_s = circmean(rad, low=-np.pi, high=np.pi)
        r_norm_s = np.linalg.norm([x_bar, y_bar])
        r_mean.append(r_mean_s)
        r_norm.append(r_norm_s)
    return r_mean, r_norm


# In[204]:


root_dir = "/media/Linux6_Data/DATA/SFB2"
proc_dir = join(root_dir, "proc")
fig_dir = join(proc_dir, "figs")


# In[205]:


ROIs = ["frontal", "parietal"]
pfs = [[12, 15], [15, 18]]


# In[206]:


for ROI in ROIs:
    ur_epo = mne.read_epochs(join(proc_dir, f"grand_{ROI}_SI-epo.fif"))
    epo = ur_epo["Index!=99"].copy() # onlyPost

    df = epo.metadata
    subjs = list(df["Subj"].unique())
    subjs.sort()
    conds = ["sham", "SOstim"]
    polarities = ["anodal", "cathodal"]
    osc_types = ["SO"]
    SI_df_dict = {"Subj":[], "OscType":[], "Cond":[], "StimType":[],
                "Polarity":[], "SI_norm":[], "SM_norm":[], "SI_mean":[], "SM_mean":[]}
    for pf in pfs:
        for subj in subjs:
            for osc in osc_types:
                for cond in conds:
                    for polarity in polarities:
                        if cond == "sham" and polarity == "cathodal":
                            continue
                        query_str = f"Subj=='{subj}' and OscType=='{osc}' and Cond=='{cond}' and Polarity=='{polarity}'"
                        
                        this_df = df.query(query_str)
                        if len(this_df) > 10:
                            SIs = this_df["SI"].values
                            SMs = this_df["Spind_Max_{}-{}Hz".format(pf[0], pf[1])].values
                            SI_mean, SI_r_norm = r_vector(SIs)
                            SM_mean, SM_r_norm = r_vector(SMs)
                            SI_df_dict["Subj"].append(subj)
                            SI_df_dict["OscType"].append(osc)
                            SI_df_dict["Cond"].append(cond)
                            SI_df_dict["StimType"].append(cond)
                            SI_df_dict["Polarity"].append(polarity)    
                            SI_df_dict["SI_norm"].append(SI_r_norm)
                            SI_df_dict["SM_norm"].append(SM_r_norm)
                            SI_df_dict["SI_mean"].append(SI_mean)
                            SI_df_dict["SM_mean"].append(SM_mean)
        SI_df = pd.DataFrame.from_dict(SI_df_dict)
        SI_df.to_csv(join(fig_dir, "r_vector_"+ROI+".csv"),index=False)
        #print(SI_df)
        for osc in osc_types:
            fig, axes = plt.subplots(len(conds),len(polarities),figsize=(38.4,21.6),
                                    subplot_kw={"projection":"polar"})
            for pol_idx, pol in enumerate(polarities):
                for cond_idx, cond in enumerate(conds):
                    query_str = f"OscType=='{osc}' and Cond=='{cond}' and Polarity=='{pol}'"
                    if cond == "sham" and pol == "cathodal":
                        continue
                        #query_str = f"OscType=='{osc}' and Cond=='{cond}'and Polarity=='anodal'"
                        
                    
                    this_df = df.query(query_str)
                    
                    this_SI_df = SI_df.query(query_str)
                    #print(f"OscType=='{osc}' and Cond=='{cond}' and Polarity=='{pol}")
                    
                    subj_spinds = this_SI_df["SI_mean"].values
                    subj_name = this_SI_df["Subj"]
                    subj_mean, subj_r_norm = r_vector(subj_spinds)
                    
                    subj_mean_list = this_SI_df["SI_mean"].values
                    subj_r_norm_list = this_SI_df["SI_norm"].values
                    
                    mean, r_norm = r_vector(this_df["Spind_Max_{}-{}Hz".format(pf[0], pf[1])].values)
                
                    vecs = [[(subj_mean, subj_r_norm), {"color":"red","linewidth":4}],
                            [(mean, r_norm), {"color":"blue","linewidth":4}]]
                   
                    for x, y in zip(subj_mean_list, subj_r_norm_list):
                       vecs.append([(x, y), {"color": "black", "linewidth": 2}])
                    
                    spind_max = this_df[f"Spind_Max_{pf[0]}-{pf[1]}Hz"]     
                          
                    circ_hist_norm(axes[pol_idx, cond_idx], spind_max.values,
                                points=subj_spinds, vecs=vecs, labels=subj_name, alpha=0.3,
                                points_col="red", bins=12)
                    axes[pol_idx,cond_idx].set_title(f"{cond} {pol}")
                    
                    

            plt.suptitle(f"Spindle ({pf[0]}-{pf[1]}Hz) Peak on {osc} phase, {ROI}")
            plt.tight_layout()
            plt.savefig(join(fig_dir, f"SI_polar_hist_{osc}_{pf[0]}-{pf[1]}Hz_{ROI}.png"))

            


        d = SI_df.query("OscType=='SO'")
        fig, ax = plt.subplots(figsize=(38.4,21.6))
        sns.barplot(data=d, x="Cond", hue="Polarity", y="SI_norm", ax=ax)
        plt.ylabel("Resultant Vector")
        plt.suptitle("Slow Oscillations")
        plt.savefig(join(fig_dir, "SI_resvec_bar_SO_all.png"))
        #formula = "SM_norm ~ C(StimType, Treatment('sham'))*C(Polarity, Treatment('anodal'))"
        #mod = smf.mixedlm(formula, data=d, groups=SI_df["Subj"])
        #mf = mod.fit(reml=False)
        #print(mf.summary())
        #formula = "SI_norm ~ C(StimType, Treatment('sham'))*C(Polarity, Treatment('anodal'))"
        #mod = smf.mixedlm(formula, data=d, groups=SI_df["Subj"])
        #mf = mod.fit(reml=False)
        #print(mf.summary())


# # for figure 1<br>
# pfs = [[12, 15], [15, 18]]<br>
# # mos_text = 
# <br>
# #            PPP<br>
# #            012<br>
# #            012<br>
# #            012<br>
# #            012<br>
# #            012<br>
# #            012<br>
# #            012<br>
# #            345<br>
# #            345<br>
# #            345<br>
# #            345<br>
# #            345<br>
# #            345<br>
# #            345<br>
#            
# <br>
# fig, axes = plt.subplot_mosaic(mos_text, subplot_kw={"projection":"polar"},<br>
#                                figsize=(21.6, 21.6))<br>
# for pf_idx, pf in enumerate(pfs):<br>
#     print("\n{}-{}Hz".format(pf[0], pf[1]))<br>
#     SI_df_dict = {"Subj":[], "Sync":[], "OscType":[], "StimType":[],<br>
#                   "SM_norm":[], "SM_mean":[]}<br>
#     for subj in subjs:<br>
#         for osc in osc_types:<br>
#             for cond in conds:<br>
#                 query_str = "Subj=='{}' and OscType=='{}' and StimType=='{}'".format(subj, osc, cond)<br>
#                 this_df = df.query(query_str)<br>
#                 if len(this_df) > 10:<br>
#                     SIs = this_df["SI"].values<br>
#                     SMs = this_df["Spind_Max_{}-{}Hz".format(pf[0], pf[1])].values<br>
#                     SI_mean, SI_r_norm = r_vector(SIs)<br>
#                     SM_mean, SM_r_norm = r_vector(SMs)<br>
#                     SI_df_dict["Subj"].append(subj)<br>
#                     SI_df_dict["Sync"].append(this_df["Sync"].iloc[0])<br>
#                     SI_df_dict["OscType"].append(osc)<br>
#                     SI_df_dict["StimType"].append(cond)<br>
#                     SI_df_dict["SM_norm"].append(SM_r_norm)<br>
#                     SI_df_dict["SM_mean"].append(SM_mean)

#     SI_df = pd.DataFrame.from_dict(SI_df_dict)

#     colors = ["blue", "red", "green"]<br>
#     cond_subj_spinds = {}<br>
#     cond_spinds = {}<br>
#     for cond_idx, cond in enumerate(conds):<br>
#         #this_ax = axes[pf_idx, cond_idx]<br>
#         this_ax = axes[str(cond_idx + pf_idx*3)]<br>
#         query_str = "OscType=='SO' and StimType=='{}'".format(cond)<br>
#         this_df = df.query(query_str)<br>
#         this_SI_df = SI_df.query(query_str)<br>
#         cond_subj_spinds[cond] = this_SI_df["SM_mean"].values<br>
#         cond_spinds[cond] = this_df["Spind_Max_{}-{}Hz".format(pf[0], pf[1])].values<br>
#         subj_mean, subj_r_norm = r_vector(cond_subj_spinds[cond])<br>
#         mean, r_norm = r_vector(this_df["Spind_Max_{}-{}Hz".format(pf[0], pf[1])].values)<br>
#         print("{}: subj_mean: {}/{} subj_r_norm: {}".format(cond, np.round(subj_mean, 3), np.round(360+np.rad2deg(subj_mean), 2), np.round(subj_r_norm, 2)))<br>
#         print("{}: mean: {}/{} r_norm: {}\n".format(cond, np.round(mean, 3), np.round(360+np.rad2deg(mean), 2), np.round(r_norm, 2)))<br>
#         vecs = [[(subj_mean, subj_r_norm), {"color":"gray","linewidth":8}],<br>
#                 [(mean, r_norm), {"color":colors[cond_idx],"linewidth":8}]]<br>
#         circ_hist_norm(this_ax, this_df["Spind_Max_{}-{}Hz".format(pf[0], pf[1])].values,<br>
#                        points=cond_subj_spinds[cond], vecs=vecs, alpha=0.3,<br>
#                        color=colors[cond_idx], points_col="gray", bins=24,<br>
#                        dot_size=280)<br>
#         this_ax.tick_params(pad=25)

# plt.suptitle("Spindle Peak on SO phase", fontsize=52)<br>
# axes["P"].axis("off")<br>
# #axes["W"].axis("off")<br>
# axes["0"].set_title("12-15Hz\n", fontsize=52)<br>
# axes["3"].set_title("15-18Hz\n", fontsize=52)<br>
# axes["3"].set_xlabel("\nSham", fontsize=52)<br>
# axes["4"].set_xlabel("\nEigen frequency", fontsize=52)<br>
# axes["5"].set_xlabel("\nFixed frequency", fontsize=52)<br>
# plt.tight_layout()<br>
# plt.savefig("../images/polar_hist_fig1_SO_{}".format(method))<br>
# plt.savefig("../images/polar_hist_fig1_SO_{}.svg".format(method))
