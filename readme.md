# Memory SFB2 Pipeline

Preprocessing and data analysis pipeline for nap eeg data in an so-tDCS study conducted at Department of Neurology, Universitätsmedizin Greifswald.
Extends previous work by Jevri Hanna published here: [https://github.com/jshanna100/sfb](https://github.com/jshanna100/sfb).

## Steps to process NAP data:
Along suffix naming conventions for input/output of each step.

### 00_00_convert: -> -raw.fif
Find raw files in raw/ folder and save to proc/ folder in .fif format.
	
### 01_00_prep: -raw -> -rp
General preprocessing: Notch filter, Resample

### 01_02_mark_stimulation_algo: -rp -> -rpa
Mark stim intervals for all three different modalities:
* sotDCS (+/- using Jevris algorithm)
* tDCS (similar technique but different events detected)
* sham (based on trigger namings)

### 01_03_cutout_raw: -rpa -> -rpac
Do *not* cutout anything (though the code to do so would be there).
Instead check if there is a concatenation point in a postim interval, in that case reject it.

### 01_04_mark_badchans: -rpac -> -rpacb
Mark bad channel using Jevris anoar script

### (01_05_detrend: -rpacb -> -rpacbd)
Not used. Detrending

### (01_06_detect_artifacts)
Not used. Unfinished script to detect artifacts (by Leo).

### **Manual** visual inspection: -rpacb -> -rpacbi
- using `helpers/inspect_files.ipynb` 

### 02_00_mark_osc: -rpacbi -> {ROI}_{osc_type}_{polarity}-epo
Find slow oscillations using Jevris algorithm.
* ROI = frontal|parietal (channels may vary according to bad markings/presence)
* osc_type = SO (not actually a variable)
* polarity = anodal|cathodal (only different for condition sotDCS_cat)

Note: There is a divergent, but more documented version of this in `legacy/02_00_mark_osc_jb.ipynb`, which also does a comparison with YASA's SO detection.
Final definition of ROIs/alternatives is a bit different though.

### 02_01_epo_cat: {ROI}_{osc_type}_{polarity}-epo -> grand-epo.fif
Combines the epo files from mark_osc into one, grand-epo.fif file (over all subjects, conditions, etc.)

The remaining steps work with the resulting `grand-epo.fif` file.

### 03_00_tfr_epo
Do a TFR analysis on the grand_epo SOs, compare anodal, cathodal, sham in a figure

### 03_01_erpac
Do an ERPAC analysis on the grand epo SOs, compare anodal, cathodal, sham in separate figures

### 03_02_erpac_sham
Use ERPAC to compare anodal and cathodal sham with each other.

### 04_00_ttest_sham
Subjects-average, paired-sample t-test of the sham conditions with other, with multiple comparisons correction

### 04_01_ttest
Subjects-average, paired-sample t-test of the anodal and cathodal stimulation against anodal sham, with multiple comparisons correction.

### 04_02_slope_est
Calculate the slope o` the PSD for sham and stim conditions.
