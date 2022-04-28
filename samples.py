# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:22:03 2021

@author: rafae
"""


import ee


def fc_random_samples_tot(img_clas_base, roi, input_scale, n_samples, n_seed, name_class_base):
    
    
   
   img_samples = img_clas_base.where(img_clas_base.lte(3), 1).clip(roi);
   
   samples = img_samples.addBands(ee.Image.pixelLonLat()).stratifiedSample(
            numPoints= n_samples, classBand= name_class_base, projection=  'EPSG:4326',scale= input_scale,region= roi, seed= n_seed)
   
   def function_point(f):
        return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))
   samples = samples.map(function_point)
   
   samples_clas = img_clas_base.reduceRegions(collection=samples, reducer= ee.Reducer.first(), scale= 30,  tileScale= 8)
   
   

    
   return samples_clas





def fc_random_samples_by_class(img_clas_base, roi, input_scale, n_samples, n_seed, name_class_base):
    
    
   samples = img_clas_base.addBands(ee.Image.pixelLonLat()).stratifiedSample(
            numPoints= n_samples, classBand= name_class_base, projection=  'EPSG:4326',scale= input_scale,region= roi, seed= n_seed)
   
   def function_point(f):
        return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))
   samples = samples.map(function_point)
   
 
    
   return samples