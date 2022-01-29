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
import itertools
from nilearn.input_data import NiftiMasker
from nilearn.image import get_data
import matplotlib.pyplot as plt
import sklearn

mouse_images = []
mouse_images_folder = ['/Users/AkhilBedapudi/Desktop/Bass_Independent_Study/All_Images/AD5x/reg_images/', '/Users/AkhilBedapudi/Desktop/Bass_Independent_Study/All_Images/AD5x/reg_images/']
mifl =list(itertools.chain.from_iterable(itertools.repeat(mouse_images_folder, 102)))
fileextension = ['_T2_to_MDT.nii.gz', '_T2_to_MDT.nii.gz']
fileextension204 = list(itertools.chain.from_iterable(itertools.repeat(fileextension, 102)))
excel_path = '/Users/AkhilBedapudi/Desktop/Bass_Independent_Study/MouseInfo.xlsx'
mouse_database = pd.read_excel(excel_path)
#genotypes = mouse_database.Genotype
#timepoints = list(mouse_database.TimePoint)
timepointstring = ['1', '2', '3','1', '2', '3','1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3' ,'1', '2', '3','1', '2', '3','1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3' , '1', '2', '3', '1', '2', '3','1', '2', '3','1', '2', '3','1', '2', '3','1', '2', '3','1', '2', '3','1', '2', '3','1', '2', '3','1', '2', '3','1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3' ]
mouse_names = list(mouse_database.Mouse)
mouse_paths = []
filepathmousenames = [''.join(z) for z in zip(mifl, mouse_names)]
filepathtimepoints = [''.join(z) for z in zip(filepathmousenames, timepointstring)]
fullfilepaths = [''.join(z) for z in zip(filepathtimepoints,fileextension204)]
gray_matter_map_filenames = fullfilepaths
timepoint = mouse_database.iloc[:,2]
genotype = mouse_database.iloc[:,3]
treatment = mouse_database.iloc[:,4]

print('First gray-matter anatomy image (3D) is located at: %s' %
      gray_matter_map_filenames[0])  # 3D data

from sklearn.model_selection import train_test_split
gm_imgs_train, gm_imgs_test, timepoint_train, timepoint_test = train_test_split(
    gray_matter_map_filenames, timepoint, train_size=.6, random_state=0)


nifti_masker = NiftiMasker(
    standardize=False,
    smoothing_fwhm=2,
    memory='nilearn_cache')  # cache options
gm_maps_masked = nifti_masker.fit_transform(gm_imgs_train)

# The features with too low between-subject variance are removed using
# :class:`sklearn.feature_selection.VarianceThreshold`.
from sklearn.feature_selection import VarianceThreshold
variance_threshold = VarianceThreshold(threshold=.01)
gm_maps_thresholded = variance_threshold.fit_transform(gm_maps_masked)

# Then we convert the data back to the mask image in order to use it for
# decoding process
mask = nifti_masker.inverse_transform(variance_threshold.get_support())


from nilearn.decoding import DecoderRegressor
decoder = DecoderRegressor(estimator='svr', mask=mask,
                           scoring='neg_mean_absolute_error',
                           screening_percentile=1,
                           n_jobs=1)
# Fit and predict with the decoder
decoder.fit(gm_imgs_train, timepoint_train)

# Sort test data for better visualization (trend, etc.)

timepointtest = pd.Series(timepoint_test)
timepoint_test = timepointtest.sort_values(ascending=False)
perm = np.argsort(timepoint_test)[::-1]
gm_imgs_test = np.array(gm_imgs_test)[perm]
timepoint_pred = decoder.predict(gm_imgs_test)

prediction_score = -np.mean(decoder.cv_scores_['beta'])

print("=== DECODER ===")
print("explained variance for the cross-validation: %f" % prediction_score)
print("")

weight_img = decoder.coef_img_['beta']

# Create the figure
from nilearn.plotting import plot_stat_map, show
bg_filename = gray_matter_map_filenames[0]
z_slice = 0
display = plot_stat_map(weight_img, bg_img=bg_filename,
                        display_mode='z', cut_coords=[z_slice])
display.title("SVM weights")
show()

plt.figure(figsize=(6, 4.5))
plt.suptitle("Decoder: Mean Absolute Error %.2f years" % prediction_score)
linewidth = 3
plt.plot(timepoint_test, label="True Time Point", linewidth=linewidth)
plt.plot(timepoint_pred, '--', c="g", label="Predicted Time Point", linewidth=linewidth)
plt.ylabel("Time Point")
plt.xlabel("subject")
plt.legend(loc="best")
plt.figure(figsize=(6, 4.5))
plt.plot(timepoint_test - timepoint_pred, label="True Time Point - predicted Time Point",
         linewidth=linewidth)
plt.xlabel("subject")
plt.legend(loc="best")

print("Massively univariate model")

gm_maps_masked = NiftiMasker().fit_transform(gray_matter_map_filenames)
data = variance_threshold.fit_transform(gm_maps_masked)

# Statistical inference
from nilearn.mass_univariate import permuted_ols
neg_log_pvals, t_scores_original_data, _ = permuted_ols(
    timepoint, data,  # + intercept as a covariate by default
    n_perm=2000,  # 1,000 in the interest of time; 10000 would be better
    verbose=1, # display progress bar
    n_jobs=1)  # can be changed to use more CPUs
signed_neg_log_pvals = neg_log_pvals * np.sign(t_scores_original_data)
signed_neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    [variance_threshold.inverse_transform(signed_neg_log_pvals)])

# Show results
threshold = -np.log10(0.1)  # 10% corrected

fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')

display = plot_stat_map(signed_neg_log_pvals_unmasked, bg_img=bg_filename,
                        threshold=threshold, cmap=plt.cm.RdBu_r,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig)
title = ('Negative $\\log_{10}$ p-values'
         '\n(Non-parametric + max-type correction)')
display.title(title, y=1.2)
n_detections = (get_data(signed_neg_log_pvals_unmasked) > threshold).sum()
print('\n%d detections' % n_detections)

show()


