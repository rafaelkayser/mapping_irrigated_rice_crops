# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:27:43 2022

@author: rafae
"""

import ee
ee.Initialize()

from model import Classification


#RANDOM FOREST CLASSIFICATION

roi = ee.FeatureCollection('users/rafaelkayser/bases_map_irrig_dez20/municipios_v143_diss')
input_scale = 30
bands = ['NDVI_gs_sd', 'EVI_gs_sd','LSWI_gs_p20','LSWI_gs_p50','LSWI_gs_p80', 'NDVI_ws_p20', 'NDVI_ws_p50', 'NDVI_ws_p80', 'mLST_gs_p20', 'mLST_gs_p50',  'mLST_gs_p80','SAR_fs_p20', 'SAR_fs_p50', 'SAR_fs_p80', 'hand']
year = 2020
img_clas_actual = ee.Image('users/rafaelkayser/30_map_irrig_sul_0621/01_base/img_clas_atual_30m_wz')  #3 classes 
path_output = 'users/rafaelkayser/tests_crop_classification/col_output_rf'
n_samples = 5000
name_band_actual = 'class'
name_asset = 'teste_clas_rf_'
ls_collection = 'C2'


#Classification.random_forest(year, img_clas_actual, roi, input_scale, n_samples, name_band_actual, bands, path_output, name_asset, ls_collection)




#CLUSTER - CROP / NO CROP CLASSIFICATION


roi = ee.FeatureCollection('users/rafaelkayser/bases_map_irrig_dez20/municipios_v143_diss')
input_scale = 30
bands = ['NDVI_gs_sd', 'EVI_gs_sd']
year_initial=2019
year_final=2020
path_output = 'users/rafaelkayser/tests_crop_classification/col_clas_mask_crops'
name_asset = 'teste2_img_mask_crop_'
ls_collection = 'C2'

Classification.cluster_mask_crop_nocrop(year_initial, year_final, bands, roi, input_scale, path_output, name_asset, ls_collection)





#CLUSTER - CROP / NO CROP CLASSIFICATION


roi = ee.FeatureCollection('users/rafaelkayser/bases_map_irrig_dez20/municipios_v143_diss')
input_scale = 30
bands = ['LSWI_gs_p20','LSWI_gs_p50','LSWI_gs_p80', 'NDVI_ws_p20', 'NDVI_ws_p50', 'NDVI_ws_p80', 'mLST_gs_p20', 'mLST_gs_p50',  'mLST_gs_p80','SAR_fs_p20', 'SAR_fs_p50', 'SAR_fs_p80', 'hand']
year_initial=2019
year_final=2020
path_output = 'users/rafaelkayser/tests_crop_classification/col_clas_mask_crops'
path_crop_mask = 'users/rafaelkayser/tests_crop_classification/col_clas_mask_crops'
name_asset = 'teste_clas_rf_'
ls_collection = 'C2'


#Classification.cluster_classify_crops(year_initial, year_final, bands, roi, input_scale, path_crop_mask, path_output, name_asset, ls_collection)

        

         




         














