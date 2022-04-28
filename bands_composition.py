# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:53:58 2021

@author: rafae
"""


import ee

from get_database import getLandsatSR_C1, getLandsatSR_C2, getSentinel1, getHand  


def fc_input_bands_composition(year, roi, bands, ls_collection):    
    
    #winter season
    start_date_w = ee.Date.fromYMD(year-1,5,1)
    end_date_w = ee.Date.fromYMD(year-1,10,1)
    
    if ls_collection == "C1":
        col_ls_ws = getLandsatSR_C1(start_date_w,end_date_w,roi)
    else:
        col_ls_ws = getLandsatSR_C2(start_date_w,end_date_w,roi)
        
        
    #growing season 
    start_date_g =  ee.Date.fromYMD(year-1,11,1)
    end_date_g =  ee.Date.fromYMD(year,4,1)
    
    if ls_collection == "C1":
        col_ls_gs = getLandsatSR_C1(start_date_g,end_date_g,roi)
    else:
        col_ls_gs = getLandsatSR_C2(start_date_g,end_date_g,roi)


    #flood season
    start_date_f = ee.Date.fromYMD(year-1,9,1)
    end_date_f = ee.Date.fromYMD(year,1,1)
    col_sar_fs = getSentinel1(start_date_f,end_date_f,roi);




   
    #mBRT  --------------------------------------------------------------------------------------------------------------
    

    

    #// Calculate regional min and max and add as image properties.
    def func_reduce_mean(image):
        mean =  image.reduceRegion(reducer= ee.Reducer.mean(), geometry=image.geometry(), bestEffort=True, scale= 30);
        return image.set('mean_brt', mean.get('T_LST'))
                        
    col_BRT_aux = col_ls_gs.select('T_LST').map(func_reduce_mean);
    
    
    #// Filter out min and max region reductions that are null.
    col_BRT_aux_filt = col_BRT_aux.filterMetadata('mean_brt', 'not_equals', None)

    #// Create BRT index.
    def func_mbrt_calc(img):
        #id_ = img.id();
        mean = img.getNumber('mean_brt');
        Index = img.subtract(mean).rename('mLST').copyProperties(img,['system:time_start','system:time_end']);
        return img.addBands(Index);
    
    col_mBRT_gs = col_BRT_aux_filt.map(func_mbrt_calc)
    


    #// COMPOSITES ---------------------------------------------------------------------------------------
    
    
    #GROWING SEASON
    img_ndvi_gs_sd = col_ls_gs.select('NDVI').reduce(ee.Reducer.stdDev()).rename('NDVI_gs_sd');
    img_evi_gs_sd = col_ls_gs.select('EVI').reduce(ee.Reducer.stdDev()).rename('EVI_gs_sd');


    img_lswi_gs_p20 = col_ls_gs.select('LSWI').reduce(ee.Reducer.percentile([20])).rename('LSWI_gs_p20');
    img_lswi_gs_p50 = col_ls_gs.select('LSWI').reduce(ee.Reducer.percentile([50])).rename('LSWI_gs_p50');
    img_lswi_gs_p80 = col_ls_gs.select('LSWI').reduce(ee.Reducer.percentile([80])).rename('LSWI_gs_p80');

    img_mLST_gs_p20 = col_mBRT_gs.select('mLST').reduce(ee.Reducer.percentile([20])).rename('mLST_gs_p20');
    img_mLST_gs_p50 = col_mBRT_gs.select('mLST').reduce(ee.Reducer.percentile([50])).rename('mLST_gs_p50');
    img_mLST_gs_p80 = col_mBRT_gs.select('mLST').reduce(ee.Reducer.percentile([80])).rename('mLST_gs_p80');
    
    
    #WINTER SEASON
    img_ndvi_ws_p20 = col_ls_ws.select('NDVI').reduce(ee.Reducer.percentile([20])).rename('NDVI_ws_p20');
    img_ndvi_ws_p50 = col_ls_ws.select('NDVI').reduce(ee.Reducer.percentile([50])).rename('NDVI_ws_p50');
    img_ndvi_ws_p80 = col_ls_ws.select('NDVI').reduce(ee.Reducer.percentile([80])).rename('NDVI_ws_p80');
  
    
    
    #FLOOD SEASON
    img_sar_fs_p20 = col_sar_fs.select('VH').reduce(ee.Reducer.percentile([20])).rename('SAR_fs_p20');
    img_sar_fs_p50 = col_sar_fs.select('VH').reduce(ee.Reducer.percentile([50])).rename('SAR_fs_p50');
    img_sar_fs_p80 = col_sar_fs.select('VH').reduce(ee.Reducer.percentile([80])).rename('SAR_fs_p80');

    

    #// HAND
    img_hand = getHand().rename('hand');

    img_clas_base = ee.Image.cat([img_ndvi_gs_sd, img_evi_gs_sd, img_hand, 
                                  img_lswi_gs_p20,img_ndvi_ws_p20,img_mLST_gs_p20,img_sar_fs_p20,
                                  img_lswi_gs_p50,img_ndvi_ws_p50,img_mLST_gs_p50,img_sar_fs_p50,
                                  img_lswi_gs_p80,img_ndvi_ws_p80,img_mLST_gs_p80,img_sar_fs_p80]).clip(roi);
    
    
    img_clas_base = img_clas_base.select(bands);
    
    
    return img_clas_base
   
   
