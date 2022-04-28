# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 23:35:41 2021

@author: rafae
"""


import ee

from samples import fc_random_samples_tot, fc_random_samples_by_class
from classification import fc_cluster_clas_original, fc_cluster_clas_with_mask, fc_random_forest, fc_majority_filter, fc_accuracy
from bands_composition import fc_input_bands_composition




class Classification:
    
    def random_forest(year, img_clas_actual, roi, input_scale, n_samples, name_band_actual, bands, path_output, input_name_asset, ls_collection):


        samples_training = fc_random_samples_by_class(img_clas_actual, roi, input_scale, n_samples, 0, name_band_actual)
        
        img_clas_base = fc_input_bands_composition(year, roi, bands, ls_collection) 
    
    
        img_clas_randomforest = fc_random_forest(year, roi, img_clas_base, samples_training, input_scale)
        
        img_clas_randomforest_mf = fc_majority_filter(img_clas_randomforest)
        img_clas_randomforest_mf = img_clas_randomforest_mf.set('year', year);
    
    
        name_asset = input_name_asset + str(year)
        task = ee.batch.Export.image.toAsset(
            image= img_clas_randomforest_mf, 
            description= name_asset,
            assetId= path_output +  '/' + name_asset,
            scale= input_scale,
            maxPixels = 9e+10 , 
            region= roi.geometry());
        task.start()


    def cluster_mask_crop_nocrop(year_initial, year_final, bands, roi, input_scale, path_output, input_name_asset, ls_collection):

        
        for year in range(year_initial, year_final+1):


            #bands = ['NDVI_gs_sd', 'EVI_gs_sd']
            nclusters = 2 #crop/ no-crop
        
            img_clas_base = fc_input_bands_composition(year, roi, bands, ls_collection) 
   
            img_clas_cluster = fc_cluster_clas_original(year, roi, img_clas_base, input_scale, 'NDVI_gs_sd', nclusters)       
            img_clas_cluster = img_clas_cluster.updateMask(img_clas_cluster.gt(0));
        
        
            img_clas_cluster = img_clas_cluster.set('year', year);
      
            name_asset = input_name_asset + str(year)
            task = ee.batch.Export.image.toAsset(
                        image= img_clas_cluster, 
                        description= name_asset,
                        assetId = path_output +  '/' + name_asset,
                        scale= input_scale,
                        maxPixels = 9e+10 , 
                        region= roi.geometry());
            task.start()
            
            
 

    def cluster_classify_crops(year_initial, year_final, bands, roi, input_scale, path_crop_mask, path_output, input_name_asset, ls_collection):


        for year in range(year_initial, year_final+1):
    
            col_class_crop = ee.ImageCollection(path_crop_mask)
            img_class_crop = col_class_crop.filterMetadata('year', 'equals', year).first();
        
            img_clas_base = fc_input_bands_composition(year, roi, bands, ls_collection) 
            img_clas_cluster_irrig = fc_cluster_clas_with_mask(year, roi, img_clas_base, input_scale, 'LSWI_gs_p80', img_class_crop)
        
        
            img_clas_cluster_mf = fc_majority_filter(img_clas_cluster_irrig)
            img_clas_cluster_mf = img_clas_cluster_mf.set('year', year);
      
    
            name_asset = input_name_asset + str(year)
            task = ee.batch.Export.image.toAsset(
                image= img_clas_cluster_mf, 
                description= name_asset,
                assetId = path_output +  '/' + name_asset,
                scale= input_scale,
                maxPixels = 9e+10 , 
                region= roi.geometry());
            task.start()
            




    def accuracy(path_class, img_clas_actual, roi, input_scale, n_samples, name_file):
        
        
        col_map_rf_final = ee.ImageCollection(path_class)
        samples_val = fc_random_samples_tot(img_clas_actual, roi, input_scale, n_samples, 30, 'class')
        tab_accuracy_output = fc_accuracy(col_map_rf_final, img_clas_actual, roi, input_scale, samples_val, 'first', 'classification')
    

        task = ee.batch.Export.table.toDrive(
            collection=tab_accuracy_output, 
            folder='output_classification', 
            description=name_file, 
            fileFormat='CSV')

        task.start()







'''



# ACURÁCIA -------------------------------------------------------------------------------------------------------------
if (module_select ==4): 


    
    

    


    


# ACURÁCIA TIME SERIES -------------------------------------------------------------------------------------------------------------
if (module_select ==5): 



    col_map_rf_final = ee.ImageCollection(path_class)
   
    df_area_calc = function_calculate_area(col_map_rf_final, roi, input_scale)
    
    

    task = ee.batch.Export.table.toDrive(
            collection=df_area_calc, 
            folder='output_accuracy_0821', 
            description='area_testes_timeseries_CL', 
            fileFormat='CSV')

    task.start()


'''