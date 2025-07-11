#!/usr/bin/env python
# coding: utf-8

# This script summarizes the output of SO parameters for each subject.

# In[ ]:


import pandas as pd 
import numpy as np

df_complete=pd.read_csv("/media/Linux6_Data/johnsm/data (input+output)/so_parameters.txt", sep="\t")

df_short=pd.DataFrame(columns=["subject","area","amount_anodal","peak_amp_anodal","trough_amp_anodal","slope_anodal",
"ptp_amp_anodal","negative_duration_anodal","positive_duration_anodal","amount_cathodal","peak_amp_cathodal","trough_amp_cathodal","slope_cathodal",
"ptp_amp_cathodal","negative_duration_cathodal","positive_duration_cathodal","amount_sham","peak_amp_sham","trough_amp_sham",
"slope_sham","ptp_amp_sham","negative_duration_sham","positive_duration_sham"])

for area in ["frontal","parietal"]:
    df=df_complete[(df_complete["k"]== area)]
    for subj in [1001,1002,1003,1004,1005,1006,1008,1011,1012,1013,1015,1020,1023,1036,1038,1042,1046,1054,1055,1056,1057,1059]:
        df_subject_anodal=df[(df["subject"] == subj)&(df["condition"] == "SOstim")&(df["polarity"] == "anodal")]
        df_subject_cathodal=df[(df["subject"] == subj)&(df["condition"] == "SOstim")&(df["polarity"] == "cathodal")]
        df_subject_sham=df[(df["subject"] == subj)&(df["condition"] == "sham")]

        if df_subject_anodal.empty:
            aa,paa,taa,sa,ptpaa,nda,pda=0,0,0,0,0,0
        else:
            aa=len(df_subject_anodal)
            paa=np.mean(df_subject_anodal["peak_amp"])
            taa=np.mean(df_subject_anodal["trough_amp"])
            sa=np.mean(df_subject_anodal["slope"])
            ptpaa=np.mean(df_subject_anodal["ptp"])
            nda=np.mean(df_subject_anodal["negative_duration"])
            pda=np.mean(df_subject_anodal["positive_duration"])

        if df_subject_cathodal.empty:
            ac,pac,tac,sc,ptpac,ndc,pdc=0,0,0,0,0,0
        else:
            ac=len(df_subject_cathodal)
            pac=np.mean(df_subject_cathodal["peak_amp"])
            tac=np.mean(df_subject_cathodal["trough_amp"])
            sc=np.mean(df_subject_cathodal["slope"])
            ptpac=np.mean(df_subject_cathodal["ptp"])
            ndc=np.mean(df_subject_cathodal["negative_duration"])
            pdc=np.mean(df_subject_cathodal["positive_duration"])

        if df_subject_sham.empty:
            amounts,pas,tas,ss,ptpas,nds,pds=0,0,0,0,0,0
        else:
            amounts=len(df_subject_sham)
            pas=np.mean(df_subject_sham["peak_amp"])
            tas=np.mean(df_subject_sham["trough_amp"])
            ss=np.mean(df_subject_sham["slope"])
            ptpas=np.mean(df_subject_sham["ptp"])
            nds=np.mean(df_subject_sham["negative_duration"])
            pds=np.mean(df_subject_sham["positive_duration"])
        
        df_short.loc[len(df_short.index)]=[subj,area,aa,paa,taa,sa,ptpaa,nda,pda,ac,pac,tac,sc,ptpac,ndc,pdc,amounts,pas,tas,ss,ptpas,nds,pds]
df_short=df_short.astype({"subject":"int"})

df_short.to_csv("/media/Linux6_Data/johnsm/data (input+output)/so_parameters_short.txt", sep = "\t", index = False)

