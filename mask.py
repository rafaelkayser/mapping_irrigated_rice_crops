# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:20:40 2021

@author: rafae
"""


import ee


def fc_masking_global_datasets(roi,year):
    
    
    gsw = ee.Image('JRC/GSW1_2/GlobalSurfaceWater');
    occurrence = gsw.select('occurrence');
    

    #// Create a water mask layer, and set the image mask so that non-water areas
    th_water = 80

    nowater_mask = occurrence.unmask(0)
    
    nowater_mask = nowater_mask.updateMask(nowater_mask.lte(th_water))
    nowater_mask = nowater_mask.where(nowater_mask.lte(th_water),1).unmask(0)
    
    #nowater_mask = nowater_mask.clip(roi)
    
    

    #// ********* mascara cobertura vegetal
    gfc = ee.Image('UMD/hansen/global_forest_change_2019_v1_7').clip(roi)
    
    forest = gfc.select('treecover2000').gt(0);
    
    forest = forest.where(forest.gt(0),1).unmask(0)
    
    gain = gfc.select('gain').eq(1).unmask(0)
    
    gain = gain.where(gain,2)
    
    loss = gfc.select('loss').eq(1).unmask(0)
    
    
    balance = loss.add(gain)
    gain_tot = balance.updateMask(balance.gte(2))
    gain_tot = gain_tot.where(gain_tot,1).unmask(0)
    
    
    
    loss_tot = balance.updateMask(balance.eq(1))
    loss_tot = loss_tot.unmask(0)
    

    forest_tot = forest.add(gain_tot).subtract(loss_tot)
    
    noforest_tot = forest_tot.updateMask(forest_tot.eq(0))
    noforest_tot = noforest_tot.where(noforest_tot.eq(0),1)
    
    noforest_tot = noforest_tot.unmask(0)
    
    
    nocrops = nowater_mask.add(noforest_tot)
    nocrops = nocrops.updateMask(nocrops.gt(0)).clip(roi)
    
    
    
    
    return noforest_tot
    



    
    
   
def fc_masking_mapbiomas(year,roi):
    


    #//Select Land Cover Map
    land_cover = ee.Image('projects/mapbiomas-workspace/public/collection5/mapbiomas_collection50_integration_v1');


    year2 = str(year);
    if (year ==2020):
        year2 = str(2019)
    


    #//selecionar o ano da imagem 
    year_band = ee.String('classification_' +year2);
    land_cover_year = land_cover.select(year_band);
  

    c01 = land_cover_year.eq(3); #forest
    c02 = land_cover_year.eq(9); #Forest Plantaon
    c03 = land_cover_year.eq(11); #wetland
    c04 = land_cover_year.eq(33); #water
    c05 = land_cover_year.eq(31); #aquaculture
    
    
    nocrops = c01.Or(c02).Or(c03).Or(c04).Or(c05)
    
    nocrops = nocrops.where(nocrops, 1)
    
    nocrops = nocrops.updateMask(nocrops.eq(0)).clip(roi)
    
    return nocrops
    
    