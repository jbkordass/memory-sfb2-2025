#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
from os import listdir
import re
import pandas as pd
from os.path import isdir, join
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
matplotlib.use('agg')


# In[ ]:


def norm_overlay(x, min=10, max=20, centre=15, xmax=4e-05):
    x = x / xmax
    x = x * (max-min)/2 + centre
    return x


# In[ ]:


def create_power_dataframe(title, tfr_stim_an_mean_power, tfr_stim_ca_mean_power, tfr_sham_an_mean_power):
    """
    Create a Pandas DataFrame from the provided power data and return it.

    Parameters:
        title (str): The title of the data.
        tfr_stim_an_mean_power (array-like): Mean power for stim_an.
        tfr_stim_ca_mean_power (array-like): Mean power for stim_ca.
        tfr_sham_an_mean_power (array-like): Mean power for sham_an.

    Returns:
        pd.DataFrame: DataFrame containing the title and mean power data.
    """
    # Create a DataFrame
    df = pd.DataFrame({

        #"subj_roi": [title] * len(tfr_stim_an_mean_power),
        "subj_roi": [title] * 1,
        "stim_an_mean_power": tfr_stim_an_mean_power,
        "stim_ca_mean_power": tfr_stim_ca_mean_power,
        "sham_an_mean_power": tfr_sham_an_mean_power,
    })
    return df


# In[ ]:


def graph(ur_epo, interval, title, ROI, vmin=-3, vmax=3):

    # interval:
    #       "all"
    #       "post"
    #       "end"

    subj = ur_epo.metadata.iloc[0]["Subj"]
    
    epo = ur_epo.copy()[f"ROI=='{ROI}'"]
    epo.pick_channels([ROI])
    tfr = tfr_morlet(epo, freqs, n_cycles, return_itc=False, average=False, output="power",
                    n_jobs=n_jobs)
    tfr.crop(-2.25, 2.25)
    tfr.apply_baseline((-2.25, -1), mode="zscore")

    # Define the frequency and time ranges
    freq_range = (12, 16)  # Frequencies in Hz
    time_range = (0.2, 0.7)  # Time in seconds
    tfr_stim_an_mean_power = -1
    tfr_stim_ca_mean_power = -1
    tfr_sham_an_mean_power = -1

    if interval == "all":
        stim_an_n = len(epo["Cond=='SOstim' and Polarity=='anodal'"])
        stim_ca_n = len(epo["Cond=='SOstim' and Polarity=='cathodal'"])
        sham_an_n = len(epo["Cond=='sham' and Polarity=='anodal'"])
        tDCS_an_n = len(epo["Cond=='tDCSstim' and Polarity=='anodal'"])
    
        # get SO for overlays
        epo.crop(-2.25, 2.25)
        line_color = "white"    
        # define layout of graph
        mos_str = """
                AABBCC
                AABBCC
                DDEEFF
                DDEEFF
                """
        fig, axes = plt.subplot_mosaic(mos_str, figsize=(38.4, 21.6))
        
        #if sham_ca_n:
        #    tfr_sham_ca = tfr["Cond=='sham' and Polarity=='cathodal'"].average()
        if stim_an_n:
            so_stim_an = norm_overlay(epo["Cond=='SOstim' and Polarity=='anodal'"].average().data.squeeze())
            tfr_stim_an = tfr["Cond=='SOstim' and Polarity=='anodal'"].average()
            # Crop the TFR data to the desired time and frequency range
            tfr_stim_an_cropped =  tfr_stim_an.copy().crop(tmin=time_range[0], tmax=time_range[1], fmin=freq_range[0], fmax=freq_range[1])
            # Average over frequencies and times
            tfr_stim_an_mean_power = np.mean(tfr_stim_an_cropped.data, axis=(1, 2))
            tfr_stim_an.plot(vmin=0, vmax=vmax, axes=axes["A"], cmap=cmap_color, tmin=-1, tmax=1)
            axes["A"].set_title(f"Stimulation Anodal ({stim_an_n})")
            axes["A"].plot(tfr_stim_an.times, so_stim_an, color=line_color)
        if sham_an_n:  
            so_sham_an = norm_overlay(epo["Cond=='sham' and Polarity=='anodal'"].average().data.squeeze())
            tfr_sham_an = tfr["Cond=='sham' and Polarity=='anodal'"].average()
            # Crop the TFR data to the desired time and frequency range
            tfr_sham_an_cropped =  tfr_sham_an.copy().crop(tmin=time_range[0], tmax=time_range[1], fmin=freq_range[0], fmax=freq_range[1])
            # Average over frequencies and times
            tfr_sham_an_mean_power = np.mean(tfr_sham_an_cropped.data, axis=(1, 2))
            tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes["B"], cmap=cmap_color, tmin=-1, tmax=1)
            axes["B"].set_title(f"Sham Anodal ({sham_an_n})")
            axes["B"].plot(tfr_sham_an.times, so_sham_an, color=line_color)
        if sham_an_n and stim_an_n:    
            (tfr_stim_an-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["C"])
            axes["C"].set_title("Stim - Sham Anodal")
            axes["C"].plot(tfr_stim_an.times, so_stim_an, color="black")
        if stim_ca_n:   
            so_stim_ca = norm_overlay(epo["Cond=='SOstim' and Polarity=='cathodal'"].average().data.squeeze())
            tfr_stim_ca = tfr["Cond=='SOstim' and Polarity=='cathodal'"].average()
            # Crop the TFR data to the desired time and frequency range
            tfr_stim_ca_cropped =  tfr_stim_ca.copy().crop(tmin=time_range[0], tmax=time_range[1], fmin=freq_range[0], fmax=freq_range[1])
            # Average over frequencies and times
            tfr_stim_ca_mean_power = np.mean(tfr_stim_ca_cropped.data, axis=(1, 2))
            tfr_stim_ca.plot(vmin=0, vmax=vmax, axes=axes["D"], cmap=cmap_color, tmin=-1, tmax=1)
            axes["D"].set_title(f"Stimulation Cathodal ({stim_ca_n})")
            axes["D"].plot(tfr_stim_ca.times, so_stim_ca, color=line_color)
        if tDCS_an_n:
            so_tDCS_an = norm_overlay(epo["Cond=='tDCSstim' and Polarity=='anodal'"].average().data.squeeze())
            tfr_tDCS_an = tfr["Cond=='tDCSstim' and Polarity=='anodal'"].average()
            
            tfr_tDCS_an.plot(vmin=0, vmax=vmax, axes=axes["E"], cmap=cmap_color, tmin=-1, tmax=1)
            axes["E"].set_title(f"tDCS Anodal ({tDCS_an_n})")
            axes["E"].plot(tfr_tDCS_an.times, so_tDCS_an, color=line_color)
        if stim_ca_n and sham_an_n:   
            (tfr_stim_ca - tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["F"])
            axes["F"].set_title("Stim Cathodal - Sham Anodal")
            axes["F"].plot(tfr_stim_ca.times, so_stim_ca, color="black")

        if not stim_an_n and not sham_an_n and not stim_ca_n and not tDCS_an_n:
            return

        plt.suptitle(title + "_all")
        plt.savefig(join(fig_dir,"subj", f"{title + "_all"}.png"))
        plt.savefig(join(fig_dir,"subj", f"{title + "_all"}.svg"))
        plt.close()
        return create_power_dataframe(title, tfr_stim_an_mean_power, tfr_stim_ca_mean_power, tfr_sham_an_mean_power)
    elif interval == "post":

        stim_an_n = len(epo["Cond=='SOstim' and Polarity=='anodal'and Index!=99"])
        stim_ca_n = len(epo["Cond=='SOstim' and Polarity=='cathodal'and Index!=99"])
        sham_an_n = len(epo["Cond=='sham' and Polarity=='anodal'and Index!=99"])
        tDCS_an_n = len(epo["Cond=='tDCSstim' and Polarity=='anodal'and Index!=99"])
    
        # get SO for overlays
        epo.crop(-2.25, 2.25)
        line_color = "white"    
        # define layout of graph
        mos_str = """
                AABBCC
                AABBCC
                DDEEFF
                DDEEFF
                """
        fig, axes = plt.subplot_mosaic(mos_str, figsize=(38.4, 21.6))
        
        #if sham_ca_n:
        #    tfr_sham_ca = tfr["Cond=='sham' and Polarity=='cathodal'"].average()
        if stim_an_n:
            so_stim_an = norm_overlay(epo["Cond=='SOstim' and Polarity=='anodal'and Index!=99"].average().data.squeeze())
            tfr_stim_an = tfr["Cond=='SOstim' and Polarity=='anodal'and Index!=99"].average()
            # Crop the TFR data to the desired time and frequency range
            tfr_stim_an_cropped =  tfr_stim_an.copy().crop(tmin=time_range[0], tmax=time_range[1], fmin=freq_range[0], fmax=freq_range[1])
            # Average over frequencies and times
            tfr_stim_an_mean_power = np.mean(tfr_stim_an_cropped.data, axis=(1, 2))  
            tfr_stim_an.plot(vmin=0, vmax=vmax, axes=axes["A"], cmap=cmap_color)
            axes["A"].set_title(f"Stimulation Anodal ({stim_an_n})")
            axes["A"].plot(tfr_stim_an.times, so_stim_an, color=line_color)
        if sham_an_n:  
            so_sham_an = norm_overlay(epo["Cond=='sham' and Polarity=='anodal'and Index!=99"].average().data.squeeze())
            tfr_sham_an = tfr["Cond=='sham' and Polarity=='anodal'and Index!=99"].average()
            # Crop the TFR data to the desired time and frequency range
            tfr_sham_an_cropped =  tfr_sham_an.copy().crop(tmin=time_range[0], tmax=time_range[1], fmin=freq_range[0], fmax=freq_range[1])
            # Average over frequencies and times
            tfr_sham_an_mean_power = np.mean(tfr_sham_an_cropped.data, axis=(1, 2))
            tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes["B"], cmap=cmap_color)
            axes["B"].set_title(f"Sham Anodal ({sham_an_n})")
            axes["B"].plot(tfr_sham_an.times, so_sham_an, color=line_color)
        if sham_an_n and stim_an_n:    
            (tfr_stim_an-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["C"])
            axes["C"].set_title("Stim - Sham Anodal")
            axes["C"].plot(tfr_stim_an.times, so_stim_an, color="black")
        if stim_ca_n:   
            so_stim_ca = norm_overlay(epo["Cond=='SOstim' and Polarity=='cathodal'and Index!=99"].average().data.squeeze())
            tfr_stim_ca = tfr["Cond=='SOstim' and Polarity=='cathodal'and Index!=99"].average()
            # Crop the TFR data to the desired time and frequency range
            tfr_stim_ca_cropped =  tfr_stim_ca.copy().crop(tmin=time_range[0], tmax=time_range[1], fmin=freq_range[0], fmax=freq_range[1])
            # Average over frequencies and times
            tfr_stim_ca_mean_power = np.mean(tfr_stim_ca_cropped.data, axis=(1, 2))
            tfr_stim_ca.plot(vmin=0, vmax=vmax, axes=axes["D"], cmap=cmap_color)
            axes["D"].set_title(f"Stimulation Cathodal ({stim_ca_n})")
            axes["D"].plot(tfr_stim_ca.times, so_stim_ca, color=line_color)
        if tDCS_an_n:
            so_tDCS_an = norm_overlay(epo["Cond=='tDCSstim' and Polarity=='anodal'and Index!=99"].average().data.squeeze())
            tfr_tDCS_an = tfr["Cond=='tDCSstim' and Polarity=='anodal'and Index!=99"].average()
            
            tfr_tDCS_an.plot(vmin=0, vmax=vmax, axes=axes["E"], cmap=cmap_color)
            axes["E"].set_title(f"tDCS Anodal ({tDCS_an_n})")
            axes["E"].plot(tfr_tDCS_an.times, so_tDCS_an, color=line_color)
        if stim_ca_n and sham_an_n:   
            (tfr_stim_ca - tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["F"])
            axes["F"].set_title("Stim Cathodal - Sham Anodal")
            axes["F"].plot(tfr_stim_ca.times, so_stim_ca, color="black")

        plt.suptitle(title + "_onlyPost")
        plt.savefig(join(fig_dir,"subj", f"{title + "_onlyPost"}.png"))
        plt.savefig(join(fig_dir,"subj", f"{title + "_onlyPost"}.svg"))
        plt.close()
        return create_power_dataframe(title, tfr_stim_an_mean_power, tfr_stim_ca_mean_power, tfr_sham_an_mean_power)
    elif interval == "end":

        stim_an_n = len(epo["Cond=='SOstim' and Polarity=='anodal'and Index==99"])
        stim_ca_n = len(epo["Cond=='SOstim' and Polarity=='cathodal'and Index==99"])
        sham_an_n = len(epo["Cond=='sham' and Polarity=='anodal'and Index==99"])
        tDCS_an_n = len(epo["Cond=='tDCSstim' and Polarity=='anodal'and Index==99"])
    
        # get SO for overlays
        epo.crop(-2.25, 2.25)
        line_color = "white"    
        # define layout of graph
        mos_str = """
                AABBCC
                AABBCC
                DDEEFF
                DDEEFF
                """
        fig, axes = plt.subplot_mosaic(mos_str, figsize=(38.4, 21.6))
        
        #if sham_ca_n:
        #    tfr_sham_ca = tfr["Cond=='sham' and Polarity=='cathodal'"].average()
        if stim_an_n:
            so_stim_an = norm_overlay(epo["Cond=='SOstim' and Polarity=='anodal'and Index==99"].average().data.squeeze())
            tfr_stim_an = tfr["Cond=='SOstim' and Polarity=='anodal'and Index==99"].average()
            # Crop the TFR data to the desired time and frequency range
            tfr_stim_an_cropped =  tfr_stim_an.copy().crop(tmin=time_range[0], tmax=time_range[1], fmin=freq_range[0], fmax=freq_range[1])
            # Average over frequencies and times
            tfr_stim_an_mean_power = np.mean(tfr_stim_an_cropped.data, axis=(1, 2))
            tfr_stim_an.plot(vmin=0, vmax=vmax, axes=axes["A"], cmap=cmap_color, tmin=-1, tmax=1)
            axes["A"].set_title(f"Stimulation Anodal ({stim_an_n})")
            axes["A"].set_xlim(-1, 1)
            axes["A"].plot(tfr_stim_an.times, so_stim_an, color=line_color)
        if sham_an_n:  
            so_sham_an = norm_overlay(epo["Cond=='sham' and Polarity=='anodal'and Index==99"].average().data.squeeze())
            tfr_sham_an = tfr["Cond=='sham' and Polarity=='anodal'and Index==99"].average()
            # Crop the TFR data to the desired time and frequency range
            tfr_sham_an_cropped =  tfr_sham_an.copy().crop(tmin=time_range[0], tmax=time_range[1], fmin=freq_range[0], fmax=freq_range[1])
            # Average over frequencies and times
            tfr_sham_an_mean_power = np.mean(tfr_sham_an_cropped.data, axis=(1, 2))
            tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes["B"], cmap=cmap_color, tmin=-1, tmax=1)
            axes["B"].set_title(f"Sham Anodal ({sham_an_n})")
            axes["B"].set_xlim(-1, 1)
            axes["B"].plot(tfr_sham_an.times, so_sham_an, color=line_color)
        if sham_an_n and stim_an_n:    
            (tfr_stim_an-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["C"])
            axes["C"].set_title("Stim - Sham Anodal")
            axes["C"].set_xlim(-1, 1)
            axes["C"].plot(tfr_stim_an.times, so_stim_an, color="black")
        if stim_ca_n:   
            so_stim_ca = norm_overlay(epo["Cond=='SOstim' and Polarity=='cathodal'and Index==99"].average().data.squeeze())
            tfr_stim_ca = tfr["Cond=='SOstim' and Polarity=='cathodal'and Index==99"].average()
            # Crop the TFR data to the desired time and frequency range
            tfr_stim_ca_cropped =  tfr_stim_ca.copy().crop(tmin=time_range[0], tmax=time_range[1], fmin=freq_range[0], fmax=freq_range[1])
            # Average over frequencies and times
            tfr_stim_ca_mean_power = np.mean(tfr_stim_ca_cropped.data, axis=(1, 2))
            tfr_stim_ca.plot(vmin=0, vmax=vmax, axes=axes["D"], cmap=cmap_color, tmin=-1, tmax=1)
            axes["D"].set_title(f"Stimulation Cathodal ({stim_ca_n})")
            axes["D"].set_xlim(-1, 1)
            axes["D"].plot(tfr_stim_ca.times, so_stim_ca, color=line_color)
        if tDCS_an_n:
            so_tDCS_an = norm_overlay(epo["Cond=='tDCSstim' and Polarity=='anodal'and Index==99"].average().data.squeeze())
            tfr_tDCS_an = tfr["Cond=='tDCSstim' and Polarity=='anodal'and Index==99"].average()
            
            tfr_tDCS_an.plot(vmin=0, vmax=vmax, axes=axes["E"], cmap=cmap_color, tmin=-1, tmax=1)
            axes["E"].set_title(f"tDCS Anodal ({tDCS_an_n})")
            axes["E"].set_xlim(-1, 1)
            axes["E"].plot(tfr_tDCS_an.times, so_tDCS_an, color=line_color)
        if stim_ca_n and sham_an_n:   
            (tfr_stim_ca - tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["F"], tmin=-1, tmax=1)
            axes["F"].set_title("Stim Cathodal - Sham Anodal")
            axes["F"].set_xlim(-1, 1)
            axes["F"].plot(tfr_stim_ca.times, so_stim_ca, color="black")
            
        plt.suptitle(title + "_onlyEnd")
        plt.savefig(join(fig_dir,"subj", f"{title + "_onlyEnd"}.png"))
        plt.savefig(join(fig_dir,"subj", f"{title + "_onlyEnd"}.svg"))
        plt.close()
        return create_power_dataframe(title, tfr_stim_an_mean_power, tfr_stim_ca_mean_power, tfr_sham_an_mean_power)
        


# In[ ]:


def graph_subjavg(ur_epo, interval, title, ROI, vmin=-2, vmax=2):
    tmin = -1
    tmax = 1
    
    epo = ur_epo.copy()[f"ROI=='{ROI}'"]
    epo.pick_channels([ROI])
    
    tfr = tfr_morlet(epo, freqs, n_cycles, return_itc=False, average=False, output="power",
                    n_jobs=n_jobs)
    tfr.crop(-2.25, 2.25)
    tfr.apply_baseline((-2.25, -1), mode="zscore")
    epo.crop(-2.25, 2.25)
    subjs = list(ur_epo.metadata["Subj"].unique())
    so_stim_an, so_stim_ca, so_sham_an, so_tDCS_an = [], [], [], []
    tfr_stim_an, tfr_stim_ca, tfr_sham_an, tfr_tDCS_an = [], [], [], []
    print("####################")
    assert isinstance(epo.metadata, pd.DataFrame)
    print(epo.metadata)
    #epo["Index==99"].plot(events=True)
    print("####################")

    if interval == "all":
        title = title + "_all"
        for subj in subjs:
            subj_epo = epo.copy()[f"Subj=='{subj}'"]
            tfr_epo = tfr.copy()[f"Subj=='{subj}'"]
            print(f"grab {subj}")
            #print(str(len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal'"])))
            #print(str(len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal'"])))
            #print(str(len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"])))
            #print(str(len(subj_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal'"])))
            if 0 in (
                len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal'"]),  
                len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal'"]),
                len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"])
                ):
                print(f"crap {subj}")
                continue

            # get SO for overlays
            if len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal'"]):
                so_stim_an.append(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal'"].average())
                tfr_stim_an.append(tfr_epo.copy()["Cond=='SOstim' and Polarity=='anodal'"].average())
            if len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal'"]):
                so_stim_ca.append(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal'"].average())
                tfr_stim_ca.append(tfr_epo.copy()["Cond=='SOstim' and Polarity=='cathodal'"].average())
            if len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"]):
                so_sham_an.append(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"].average())
                tfr_sham_an.append(tfr_epo.copy()["Cond=='sham' and Polarity=='anodal'"].average())
            if len(subj_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal'"]):
                so_tDCS_an.append(subj_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal'"].average())
                tfr_tDCS_an.append(tfr_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal'"].average())
        
    elif interval == "post":
        title = title + "_onlyPost"
        for subj in subjs:
            subj_epo = epo.copy()[f"Subj=='{subj}'"]
            tfr_epo = tfr.copy()[f"Subj=='{subj}'"]
            print(f"grab {subj}")
            #print(str(len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal'"])))
            #print(str(len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal'"])))
            #print(str(len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"])))
            #print(str(len(subj_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal'"])))
            if 0 in (
                len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal'"]),  
                len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal'"]),
                len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"])
                ):
                print(f"crap {subj}")
                continue

            # get SO for overlays
            if len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal' and Index!=99"]):
                so_stim_an.append(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal' and Index!=99"].average())
                tfr_stim_an.append(tfr_epo.copy()["Cond=='SOstim' and Polarity=='anodal' and Index!=99"].average())
            if len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal' and Index!=99"]):
                so_stim_ca.append(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal' and Index!=99"].average())
                tfr_stim_ca.append(tfr_epo.copy()["Cond=='SOstim' and Polarity=='cathodal' and Index!=99"].average())
            if len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal' and Index!=99"]):
                so_sham_an.append(subj_epo.copy()["Cond=='sham' and Polarity=='anodal' and Index!=99"].average())
                tfr_sham_an.append(tfr_epo.copy()["Cond=='sham' and Polarity=='anodal' and Index!=99"].average())
            if len(subj_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal' and Index!=99"]):
                so_tDCS_an.append(subj_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal' and Index!=99"].average())
                tfr_tDCS_an.append(tfr_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal' and Index!=99"].average())

    elif interval == "end":
        title = title + "_onlyEnd"
        for subj in subjs:
            subj_epo = epo.copy()[f"Subj=='{subj}'"]
            tfr_epo = tfr.copy()[f"Subj=='{subj}'"]
            print(f"grab {subj}" + "_onlyEnd")
            #print(str(len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal'"])))
            #print(str(len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal'"])))
            #print(str(len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"])))
            #print(str(len(subj_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal'"])))
            if 0 in (
                len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal'"]),  
                len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal'"]),
                len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"])
                ):
                print(f"crap {subj}")
                continue

            # get SO for overlays
            if len(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal' and Index==99"]):
                so_stim_an.append(subj_epo.copy()["Cond=='SOstim' and Polarity=='anodal' and Index==99"].average())
                tfr_stim_an.append(tfr_epo.copy()["Cond=='SOstim' and Polarity=='anodal' and Index==99"].average())
            if len(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal' and Index==99"]):
                so_stim_ca.append(subj_epo.copy()["Cond=='SOstim' and Polarity=='cathodal' and Index==99"].average())
                tfr_stim_ca.append(tfr_epo.copy()["Cond=='SOstim' and Polarity=='cathodal' and Index==99"].average())
            if len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'and Index==99"]):
                so_sham_an.append(subj_epo.copy()["Cond=='sham' and Polarity=='anodal' and Index==99"].average())
                tfr_sham_an.append(tfr_epo.copy()["Cond=='sham' and Polarity=='anodal' and Index==99"].average())
            if len(subj_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal'and Index==99"]):
                so_tDCS_an.append(subj_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal' and Index==99"].average())
                tfr_tDCS_an.append(tfr_epo.copy()["Cond=='tDCSstim' and Polarity=='anodal' and Index==99"].average())

    so_stim_an = norm_overlay(mne.grand_average(so_stim_an).data.squeeze())
  
    so_stim_ca = norm_overlay(mne.grand_average(so_stim_ca).data.squeeze())
    so_sham_an = norm_overlay(mne.grand_average(so_sham_an).data.squeeze())
    so_tDCS_an = norm_overlay(mne.grand_average(so_tDCS_an).data.squeeze())

    stim_an_n = len(tfr_stim_an)
    stim_ca_n = len(tfr_stim_ca)
    sham_an_n = len(tfr_sham_an)
    tDCS_an_n = len(tfr_tDCS_an)
    print(str(stim_an_n))
    print(str(stim_ca_n))
    print(str(sham_an_n))
    print(str(tDCS_an_n))

    tfr_stim_an = mne.grand_average(tfr_stim_an)
    tfr_stim_ca = mne.grand_average(tfr_stim_ca)
    tfr_sham_an = mne.grand_average(tfr_sham_an)
    tfr_tDCS_an = mne.grand_average(tfr_tDCS_an)

    line_color = "white"
    # define layout of graph
    mos_str = """
            AABBCC
            AABBCC
            DDEEFF
            DDEEFF
            """
   
    fig, axes = plt.subplot_mosaic(mos_str, figsize=(38.4, 21.6))

    tfr_stim_an.plot(vmin=0, vmax=vmax, axes=axes["A"], cmap=cmap_color, tmin=-1, tmax=1)
    axes["A"].set_title(f"Stimulation Anodal ({stim_an_n})")
    times = tfr_stim_an.times
    time_mask = (times >= tmin) & (times <= tmax)
    times_selected = times[time_mask]
    so_selected = so_stim_an[time_mask]
    axes["A"].set_xlim(tmin, tmax)
    axes["A"].set_xlabel("Time (s)", fontsize=16)
    axes["A"].set_ylabel("Frequency (Hz)", fontsize=16)
    axes["A"].tick_params(axis='both', labelsize=14)
    axes["A"].plot(times_selected, so_selected, color=line_color)
    
    tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes["B"], cmap=cmap_color, tmin=-1, tmax=1)
    axes["B"].set_title(f"Sham Anodal ({sham_an_n})")
    times = tfr_sham_an.times
    time_mask = (times >= tmin) & (times <= tmax)
    times_selected = times[time_mask]
    so_selected = so_sham_an[time_mask]
    axes["B"].set_xlim(tmin, tmax)
    axes["B"].set_xlabel("Time (s)", fontsize=16)
    axes["B"].set_ylabel("Frequency (Hz)", fontsize=16)
    axes["B"].tick_params(axis='both', labelsize=14)
    axes["B"].plot(times_selected, so_selected, color=line_color)
   

    (tfr_stim_an-tfr_sham_an).plot(vmin=vmin/2+deltaV, vmax=vmax/2-deltaV, cmap=cmap_color, axes=axes["C"])
    axes["C"].set_title("Stim Anodal - Sham Anodal")
    times = tfr_stim_an.times
    time_mask = (times >= tmin) & (times <= tmax)
    times_selected = times[time_mask]
    so_selected = so_stim_an[time_mask]
    axes["C"].set_xlim(tmin, tmax)
    axes["C"].set_xlabel("Time (s)", fontsize=16)
    axes["C"].set_ylabel("Frequency (Hz)", fontsize=16)
    axes["C"].tick_params(axis='both', labelsize=14)
    axes["C"].plot(times_selected, so_selected, color="black")
    

    tfr_stim_ca.plot(vmin=0, vmax=vmax, axes=axes["D"], cmap=cmap_color, tmin=-1, tmax=1)
    axes["D"].set_title(f"Stimulation Cathodal ({stim_ca_n})")
    times = tfr_stim_ca.times
    time_mask = (times >= tmin) & (times <= tmax)
    times_selected = times[time_mask]
    so_selected = so_stim_ca[time_mask]
    axes["D"].set_xlim(tmin, tmax)
    axes["D"].set_xlabel("Time (s)", fontsize=16)
    axes["D"].set_ylabel("Frequency (Hz)", fontsize=16)
    axes["D"].tick_params(axis='both', labelsize=14)
    axes["D"].plot(times_selected, so_selected, color=line_color)
    

    tfr_tDCS_an.plot(vmin=0, vmax=vmax, axes=axes["E"], cmap=cmap_color, tmin=-1, tmax=1)
    axes["E"].set_title(f"tDCS Anodal ({tDCS_an_n})")
    times = tfr_tDCS_an.times
    time_mask = (times >= tmin) & (times <= tmax)
    times_selected = times[time_mask]
    so_selected = so_stim_ca[time_mask]
    axes["E"].set_xlim(tmin, tmax)
    axes["E"].set_xlabel("Time (s)", fontsize=16)
    axes["E"].set_ylabel("Frequency (Hz)", fontsize=16)
    axes["E"].tick_params(axis='both', labelsize=14)
    axes["E"].plot(times_selected, so_selected, color=line_color)

    (tfr_stim_ca - tfr_sham_an).plot(vmin=vmin/2+deltaV, vmax=vmax/2-deltaV, axes=axes["F"], cmap=cmap_color, tmin=-1, tmax=1)
    axes["F"].set_title("Stim Cathodal - Sham Anodal")

    times = tfr_stim_ca.times
    time_mask = (times >= tmin) & (times <= tmax)
    times_selected = times[time_mask]
    so_selected = so_stim_ca[time_mask]

    axes["F"].set_xlim(tmin, tmax)
    axes["F"].set_xlabel("Time (s)", fontsize=16)
    axes["F"].set_ylabel("Frequency (Hz)", fontsize=16)
    axes["F"].tick_params(axis='both', labelsize=14)
    axes["F"].plot(times_selected, so_selected, color="black")
    

    plt.suptitle(title)
    plt.savefig(join(fig_dir, f"{title}.png"))
    plt.savefig(join(fig_dir, f"{title}.svg"))
    plt.close()


# In[ ]:


root_dir = "/media/Linux6_Data/DATA/SFB2"
proc_dir = join(root_dir, "proc")
fig_dir = join(proc_dir, "figs")


# In[ ]:


freqs = np.linspace(10, 20, 50)
n_cycles = 5
n_jobs = 24
cmap_color = 'RdBu_r'
deltaV = 0.2


# In[ ]:


overwrite = True

ur_epo = mne.read_epochs(join(proc_dir, f"grand-epo.fif"))
ur_epo = ur_epo["OscType=='SO'"]
subjs = list(ur_epo.metadata["Subj"].unique())

ROIs = list(ur_epo.metadata["ROI"].unique())

intervals = ["all", "post", "end"]
# Initialize an empty list to store the individual DataFrames
mean_power_list = []
exclude = [
        #"1069_all",         #not checked why, seems to be empty
        #"1069_post",
        #"1069_end",
        #"1074_all",
        #"1074_post",
        #"1074_end"
        #"1015_all",
        #"1015_post",
        #"1015_end",
        ]
for ivals in intervals:
    for ROI in ROIs:
       
        graph_subjavg(ur_epo,ivals, f"subj avg {ROI}", ROI)
        for subj in subjs:
            if subj+"_"+ivals not in exclude:
                print("processing: " + subj + "_" + ivals)
                subj_epo = ur_epo.copy()[f"Subj=='{subj}'"]
                #try:
                mean_power_list.append(graph(subj_epo,ivals, f"{subj} {ROI}", ROI, vmin=-6, vmax=6))
                print(mean_power_list)                   
                #except:
                #    print("!!! Problems in " + subj + " " + ROI)    
final_mean_power_df = pd.concat(mean_power_list, ignore_index=True)      
final_mean_power_df.to_csv(join(fig_dir,"final_dataframe.csv"), index=False)    


# In[ ]:





# In[ ]:




