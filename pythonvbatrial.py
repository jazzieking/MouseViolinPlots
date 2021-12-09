#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 15:33:57 2021

@author: AkhilBedapudi
"""
import numpy as np
import gzip 
import os, glob
import pandas as pd

n_subjects = 100
#from nilearn import datasets
# oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps()
age = oasis_dataset.ext_vars['age'].astype(float)
mouse_images = []
mouse_images_folder = '/Users/akhilbedapudinew/Desktop/Bass_Independent_Study/All_Images/AD5x/reg_images/'
mouse_images = glob.glob(os.path.join(mouse_images_folder,'*.nii.gz'))
###############################################################################
# Sex is encoded as 'M' or 'F'. Hence, we make it a binary variable.

excel_path = '/Users/akhilbedapudinew/Desktop/Bass_Independent_Study/MouseInfo.xlsx'
mouse_database = pd.read_excel(excel_path)
mouse_database.add('FilePath')
genotypes = mouse_database.Genotype
timepoints = list(mouse_database.TimePoint)
mouse_names = list(mouse_database.Mouse)
mouse_paths = []
i=0
for mouse_name in mouse_names:
    mouse_path = os.path.join(mouse_images_folder,mouse_name+timepoints[i]+'_T2_to_MDT.nii.gz')
    i+1
    
pd.Series(mouse_path)
sex = oasis_dataset.ext_vars['mf'] == 'F'

###############################################################################
# Print basic information on the dataset.
#print('First gray-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.gray_matter_maps[0])  # 3D data
#print('First white-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.white_matter_maps[0])  # 3D data

###############################################################################
# Get a mask image: A mask of the  cortex of the ICBM template.
gm_mask = datasets.fetch_icbm152_brain_gm_mask()

###############################################################################
# Resample the images, since this mask has a different resolution.
from nilearn.image import resample_to_img
mask_img = resample_to_img(
    gm_mask, gray_matter_map_filenames[0], interpolation='nearest')

#############################################################################
# Analyse data
# ------------
#
# First, we create an adequate design matrix with three columns: 'age',
# 'sex', 'intercept'.
import pandas as pd
import numpy as np
intercept = np.ones(n_subjects)
design_matrix = pd.DataFrame(np.vstack((age, sex, intercept)).T,
                             columns=['age', 'sex', 'intercept'])

#############################################################################
# Let's plot the design matrix.
from nilearn.plotting import plot_design_matrix

ax = plot_design_matrix(design_matrix)
ax.set_title('Second level design matrix', fontsize=12)
ax.set_ylabel('maps')

##########################################################################
# Next, we specify and fit the second-level model when loading the data and
# also smooth a little bit to improve statistical behavior.

from nilearn.glm.second_level import SecondLevelModel
second_level_model = SecondLevelModel(smoothing_fwhm=2.0, mask_img=mask_img)
second_level_model.fit(gray_matter_map_filenames,
                       design_matrix=design_matrix)

##########################################################################
# Estimating the contrast is very simple. We can just provide the column
# name of the design matrix.
z_map = second_level_model.compute_contrast(second_level_contrast=[1, 0, 0],
                                            output_type='z_score')

###########################################################################
# We threshold the second level contrast at uncorrected p < 0.001 and plot it.
from nilearn import plotting
from nilearn.glm import threshold_stats_img
_, threshold = threshold_stats_img(
    z_map, alpha=.05, height_control='fdr')
print('The FDR=.05-corrected threshold is: %.3g' % threshold)

display = plotting.plot_stat_map(
    z_map, threshold=threshold, colorbar=True, display_mode='z',
    cut_coords=[-4, 26],
    title='age effect on grey matter density (FDR = .05)')
plotting.show()

###########################################################################
# We can also study the effect of sex by computing the contrast, thresholding
# it and plot the resulting map.

z_map = second_level_model.compute_contrast(second_level_contrast='sex',
                                            output_type='z_score')
_, threshold = threshold_stats_img(
    z_map, alpha=.05, height_control='fdr')
plotting.plot_stat_map(
    z_map, threshold=threshold, colorbar=True,
    title='sex effect on grey matter density (FDR = .05)')

###########################################################################
# Note that there does not seem to be any significant effect of sex on
# grey matter density on that dataset.

###########################################################################
# Generating a report
# -------------------
# It can be useful to quickly generate a
# portable, ready-to-view report with most of the pertinent information.
# This is easy to do if you have a fitted model and the list of contrasts,
# which we do here.

from nilearn.reporting import make_glm_report

icbm152_2009 = datasets.fetch_icbm152_2009()
report = make_glm_report(model=second_level_model,
                         contrasts=['age', 'sex'],
                         bg_img=icbm152_2009['t1'],
                         )

#########################################################################
# We have several ways to access the report:

# report  # This report can be viewed in a notebook
# report.save_as_html('report.html')
# report.open_in_browser()
