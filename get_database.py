# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 23:01:10 2021

@author: rafae
"""

import ee


## LANDSAR SR


def getLandsatSR_C2(startDate,endDate,roi):

    col_SR_L8 =(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                        .filterDate(startDate,endDate)
                        .filterBounds(roi)
                        .filterMetadata('CLOUD_COVER', 'less_than', 50)
                        .map(applyScaleFactors_L8)
                        .select([0,1,2,3,4,5,6,8,17],["UB","B","G","R","NIR","SWIR_1", "SWIR_2", "BRT", "QA_PIXEL"])
                        .map(cloud_mask_C2_l8)
                        .map(f_albedoL8_C2)
                        .map(addSI_landsat)
                        .map(calc_LST_C2)
                        )  

    col_SR_L5 =(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
                        .filterDate(startDate,endDate)
                        .filterBounds(roi)
                        .filterMetadata('CLOUD_COVER', 'less_than', 50)
                        .map(applyScaleFactors_L5L7)
                        .select([0,1,2,3,4,5,8,17], ["B","G","R","NIR","SWIR_1","SWIR_2","BRT", "QA_PIXEL"])
                        .map(cloud_mask_C2_l457)                     
                        .map(f_albedoL5L7_C2)
                        .map(addSI_landsat)
                        .map(calc_LST_C2)
                        );

    col_SR_L7 =(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                        .filterDate(startDate,endDate)
                        .filterBounds(roi)
                        .filterMetadata('CLOUD_COVER', 'less_than', 50)
                        .map(applyScaleFactors_L5L7)
                        .select([0,1,2,3,4,5,8,17], ["B","G","R","NIR","SWIR_1","SWIR_2","BRT", "QA_PIXEL"])
                        .map(cloud_mask_C2_l457)            
                        .map(f_albedoL5L7_C2)
                        .map(addSI_landsat)
                        .map(calc_LST_C2)                                               
                        );
    
    return ee.ImageCollection(col_SR_L5.merge(col_SR_L7).merge(col_SR_L8));
    
   
    

def applyScaleFactors_L8(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)


def applyScaleFactors_L5L7(image):
  opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, None, True).addBands(thermalBand, None, True)


def cloud_mask_C2_l457(landsat_image):
    """Cloud mask (Landsat 4/5/7)"""

    quality = landsat_image.select('QA_PIXEL')
    c01 = quality.eq(5440)  # Clear (0001010101000000)
    c02 = quality.eq(5504)  # Water (0001010110000000)
    mask = c01.Or(c02)

    return landsat_image.updateMask(mask);


def cloud_mask_C2_l8(landsat_image):
    """Cloud mask (Landsat 8)"""

    quality = landsat_image.select('QA_PIXEL')
    c01 = quality.eq(21824)  # Clear (101010101000000)
    c02 = quality.eq(21952)  # Water (101010111000000)
    c03 = quality.eq(1346)   # ?     (000010101000010)
    mask = c01.Or(c02).Or(c03)

    return landsat_image.updateMask(mask);    



#ALBEDO
def f_albedoL5L7_C2(image):
    
    alfa = image.expression(
      '(0.254*B1) + (0.149*B2) + (0.147*B3) + (0.311*B4) + (0.103*B5) + (0.036*B7)',{
        'B1' : image.select(['B']),#.divide(10000),
        'B2' : image.select(['G']),#.divide(10000),
        'B3' : image.select(['R']),#.divide(10000),
        'B4' : image.select(['NIR']),#.divide(10000),
        'B5' : image.select(['SWIR_1']),#.divide(10000),
        'B7' : image.select(['SWIR_2'])#.divide(10000)
      }).rename('ALFA');
    return image.addBands(alfa);

def f_albedoL8_C2(image):
    alfa = image.expression(
      '(0.130*B1) + (0.115*B2) + (0.143*B3) + (0.180*B4) + (0.281*B5) + (0.108*B6) + (0.042*B7)',{  #// (Ke, Im  et al 2016)
        'B1' : image.select(['UB']),#.divide(10000),
        'B2' : image.select(['B']),#.divide(10000),
        'B3' : image.select(['G']),#.divide(10000),
        'B4' : image.select(['R']),#.divide(10000),
        'B5' : image.select(['NIR']),#.divide(10000),
        'B6' : image.select(['SWIR_1']),#.divide(10000),
        'B7' : image.select(['SWIR_2'])#.divide(10000)
      }).rename('ALFA');
    return image.addBands(alfa);


###############################################################################################################################################################




def getLandsatSR_C1(startDate,endDate,roi):

    col_SR_L8 =(ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
                        .filterDate(startDate,endDate)
                        .filterBounds(roi)
                        .select([0,1,2,3,4,5,7,6,10],["UB","B","G","R","NIR","SWIR_1", "BRT","SWIR_2","pixel_qa"])
                        .map(f_cloudMaskL8_SR)
                        .filterMetadata('CLOUD_COVER', 'less_than', 50)
                        .map(f_albedoL8_C1)
                        .map(addSI_landsat)
                        .map(calc_LST_C1)
                        )  

    col_SR_L5 =(ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
                        .filterDate(startDate,endDate)
                        .filterBounds(roi)
                        .select([0,1,2,3,4,5,6,9], ["B","G","R","NIR","SWIR_1","BRT","SWIR_2", "pixel_qa"])
                        .map(f_cloudMaskL457_SR)
                        .filterMetadata('CLOUD_COVER', 'less_than', 50)
                        .map(f_albedoL5L7_C1)
                        .map(addSI_landsat)
                        .map(calc_LST_C1)
                        )  

    col_SR_L7 =(ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')
                        .filterDate(startDate,endDate)
                        .filterBounds(roi)                     
                        .select([0,1,2,3,4,5,6,9], ["B","G","R","NIR","SWIR_1","BRT","SWIR_2", "pixel_qa"])
                        .map(f_cloudMaskL457_SR)
                        .filterMetadata('CLOUD_COVER', 'less_than', 50)
                        .map(f_albedoL5L7_C1)
                        .map(addSI_landsat)
                        .map(calc_LST_C1)
                        )  
    
    return ee.ImageCollection(col_SR_L5.merge(col_SR_L7).merge(col_SR_L8));
    
    
    
#Function to mask clouds in Landsat 5/7 SR imagery.
def f_cloudMaskL457_SR(image):
    
        quality = image.select('pixel_qa');
        c01 = quality.eq(66);# //Clear, low confidence cloud
        c02 = quality.eq(68); #//water, low confidence cloud
        mask = c01.Or(c02);
        return image.updateMask(mask);

def f_cloudMaskL8_SR(image):
        quality = image.select('pixel_qa');
        c01 = quality.eq(322); #//Clear, low confidence cloud
        c02 = quality.eq(324); #//water, low confidence cloud
        c03 = quality.eq(1346);# //Clear terrain, terrain occluded
        mask = c01.Or(c02).Or(c03);
        return image.updateMask(mask);
    


#ALBEDO
def f_albedoL5L7_C1(image):
    
    alfa = image.expression(
      '(0.254*B1) + (0.149*B2) + (0.147*B3) + (0.311*B4) + (0.103*B5) + (0.036*B7)',{
        'B1' : image.select(['B']).divide(10000),
        'B2' : image.select(['G']).divide(10000),
        'B3' : image.select(['R']).divide(10000),
        'B4' : image.select(['NIR']).divide(10000),
        'B5' : image.select(['SWIR_1']).divide(10000),
        'B7' : image.select(['SWIR_2']).divide(10000)
      }).rename('ALFA');
    return image.addBands(alfa);

def f_albedoL8_C1(image):
    alfa = image.expression(
      '(0.130*B1) + (0.115*B2) + (0.143*B3) + (0.180*B4) + (0.281*B5) + (0.108*B6) + (0.042*B7)',{  #// (Ke, Im  et al 2016)
        'B1' : image.select(['UB']).divide(10000),
        'B2' : image.select(['B']).divide(10000),
        'B3' : image.select(['G']).divide(10000),
        'B4' : image.select(['R']).divide(10000),
        'B5' : image.select(['NIR']).divide(10000),
        'B6' : image.select(['SWIR_1']).divide(10000),
        'B7' : image.select(['SWIR_2']).divide(10000)
      }).rename('ALFA');
    return image.addBands(alfa);




def getHand():
  return ee.Image('users/gena/GlobalHAND/30m/hand-1000');



def getSentinel1(startDate,endDate,roi):
    
    
    cols1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
            .filterBounds(roi)
            .filterDate(startDate,endDate)
            .select(['VH'])
            .map(RefinedLee))
    
    return cols1
            
            

def getS1_DESC(img):
	VH = img.select(['VH'])	
	VV = img.select(['VV'])
	angle = img.select(['angle'])
	return img.select([]).addBands([VH,VV,angle]).select([0,1,2],['VH','VV','angle'])  


#Function to convert from dB
def toNatural(img):
	return ee.Image(10.0).pow(img.select(0).divide(10.0));


#Function to convert to dB
def toDB(img):
	return ee.Image(img).log10().multiply(10.0);

#Apllying a Refined Lee Speckle filter as coded in the SNAP 3.0 S1TBX:
#https:#github.com/senbox-org/s1tbx/blob/master/s1tbx-op-sar-processing/src/main/java/org/esa/s1tbx/sar/gpf/filtering/SpeckleFilters/RefinedLee.java
def RefinedLee(img):
  # img must be in natural units, i.e. not in dB!
  # Set up 3x3 kernels
   
  # convert to natural.. do not apply function on dB!
  myimg = toNatural(img);
   
  weights3 = ee.List.repeat(ee.List.repeat(1,3),3);
  kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False);
   
  mean3 = myimg.reduceNeighborhood(ee.Reducer.mean(), kernel3);
  variance3 = myimg.reduceNeighborhood(ee.Reducer.variance(), kernel3);
   
  # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions

  sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]]);
 
  sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False);
   
  # Calculate mean and variance for the sampled windows and store as 9 bands
  sample_mean = mean3.neighborhoodToBands(sample_kernel);
  sample_var= variance3.neighborhoodToBands(sample_kernel);
   
  # Determine the 4 gradients for the sampled windows
  gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
  gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
  gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
  gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());
   
  # And find the maximum gradient amongst gradient bands
  max_gradient = gradients.reduce(ee.Reducer.max());
   
  # Create a mask for band pixels that are the maximum gradient
  gradmask = gradients.eq(max_gradient);
   
  # duplicate gradmask bands: each gradient represents 2 directions
  gradmask = gradmask.addBands(gradmask);
   
  # Determine the 8 directions
  directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
  directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
  directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
  directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
  # The next 4 are the not() of the previous 4
  directions = directions.addBands(directions.select(0).Not().multiply(5));
  directions = directions.addBands(directions.select(1).Not().multiply(6));
  directions = directions.addBands(directions.select(2).Not().multiply(7));
  directions = directions.addBands(directions.select(3).Not().multiply(8));
   
  # Mask all values that are not 1-8
  directions = directions.updateMask(gradmask);
   
  # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
  directions = directions.reduce(ee.Reducer.sum());
   
  sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));
   
  # Calculate localNoiseVariance
  sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0]);
   
  # Set up the 7*7 kernels for directional statistics
  rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4));

  diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0],
  [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]);
   
  rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False);
  diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False);
   
  # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
  dir_mean = myimg.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
  dir_var = myimg.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));
   
  dir_mean = dir_mean.addBands(myimg.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
  dir_var= dir_var.addBands(myimg.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));
 
  # and add the bands for rotated kernels
  for i in range(1, 4):
	  dir_mean = dir_mean.addBands(myimg.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
	  dir_var = dir_var.addBands(myimg.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
	  dir_mean = dir_mean.addBands(myimg.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
	  dir_var = dir_var.addBands(myimg.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
	  
   
  # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
  dir_mean = dir_mean.reduce(ee.Reducer.sum());
  dir_var = dir_var.reduce(ee.Reducer.sum());
   
  # A finally generate the filtered value
  varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0));
   
  b = varX.divide(dir_var);
   
  result = dir_mean.add(b.multiply(myimg.subtract(dir_mean)));
  #return(result);

  return(img.select([]).addBands(ee.Image(toDB(result.arrayGet(0))).rename("VH")));





#// ADD SPECTRAL INDICES - SENTINEL
def addSI_sentinel(image):
    ndvi =  image.normalizedDifference(['NIR', 'R']).rename('NDVI')
    lswi =  image.normalizedDifference(['NIR', 'SWIR_1']).rename('LSWI')
    
    
    # //EVI
    evi = image.expression('2.5 * ((N - R) / (N + (6 * R) - (7.5 * B) + 1))', {'N': image.select('NIR'), 'R': image.select('R'), 'B': image.select('B') }).rename('EVI');
    
    
    return image.addBands([ndvi, lswi, evi]).float()


#################################################################################################################################################


#// add spectral indices band function
def addSI_landsat(image):
    ndvi =  image.normalizedDifference(['NIR', 'R']).rename('NDVI')
    lswi = image.normalizedDifference(['NIR', 'SWIR_1']).rename('LSWI')
    
    evi = image.expression('2.5 * ((N - R) / (N + (6 * R) - (7.5 * B) + 1))', {'N': image.select('NIR'), 'R': image.select('R'), 'B': image.select('B') }).rename('EVI');
 
    return image.addBands([ndvi, lswi, evi]).float()





#mBRT  --------------------------------------------------------------------------------------------------------------
    
def calc_LST_C2(image):
    
    ndvi =  image.normalizedDifference(['NIR', 'R']).rename('NDVI')
    
    NDVI_adjust=(ndvi.clamp(0.0, 1.00))
    
    fipar = (NDVI_adjust.multiply(1).subtract(ee.Number(0.05))).rename('fipar')
    fipar = fipar.clamp(0,1)
    
    #LAI JPL
    lai =image.expression(
    '-log(1 - fIPAR)/(KPAR)',{
        'fIPAR': fipar,
        'KPAR': ee.Number(0.5)
       }).rename('LAI')
    
    
    #broad-band surface emissivity
    e_0 = image.expression(
      '0.95 + 0.01 * LAI',{
        'LAI': lai});
    e_0 = e_0.where(lai.gt(3), 0.98).rename('e_0');
    
    e_0 = e_0.where(ndvi.lt(0).And(image.select('ALFA').lt(0.47)), 0.985).rename('e_0');
    e_0 = e_0.where(ndvi.lt(0).And(image.select('ALFA').gte(0.47)), 0.985).rename('e_0');
    
    #Narrow band transmissivity
    e_NB = image.expression(
      '0.97 + (0.0033 * LAI)',{'LAI': lai});
    e_NB = e_NB.where(lai.gt(3), 0.98).rename('e_NB');
    
    
    e_NB = e_NB.where(ndvi.lt(0).And(image.select('ALFA').lt(0.47)), 0.99).rename('e_NB');
    e_NB = e_NB.where(ndvi.lt(0).And(image.select('ALFA').gte(0.47)), 0.99).rename('e_NB');
    
    log_eNB = e_NB.log();   
    #LLand Surface Temperature
    comp_onda = ee.Number(1.115e-05);        
    lst = image.expression(
      'Tb / ( 1+ ( ( comp_onda * Tb / fator) * log_eNB))',{
        'Tb': image.select('BRT'),#.divide(10),
        'comp_onda': comp_onda,
        'log_eNB': log_eNB,
        'fator': ee.Number(1.438e-02),
      }).rename('T_LST');
    
    
    return image.addBands([lst]) 





def calc_LST_C1(image):
    
    ndvi =  image.normalizedDifference(['NIR', 'R']).rename('NDVI')
    
    NDVI_adjust=(ndvi.clamp(0.0, 1.00))
    
    fipar = (NDVI_adjust.multiply(1).subtract(ee.Number(0.05))).rename('fipar')
    fipar = fipar.clamp(0,1)
    
    #LAI JPL
    lai =image.expression(
    '-log(1 - fIPAR)/(KPAR)',{
        'fIPAR': fipar,
        'KPAR': ee.Number(0.5)
       }).rename('LAI')
    
    
    #broad-band surface emissivity
    e_0 = image.expression(
      '0.95 + 0.01 * LAI',{
        'LAI': lai});
    e_0 = e_0.where(lai.gt(3), 0.98).rename('e_0');
    
    e_0 = e_0.where(ndvi.lt(0).And(image.select('ALFA').lt(0.47)), 0.985).rename('e_0');
    e_0 = e_0.where(ndvi.lt(0).And(image.select('ALFA').gte(0.47)), 0.985).rename('e_0');
    
    #Narrow band transmissivity
    e_NB = image.expression(
      '0.97 + (0.0033 * LAI)',{'LAI': lai});
    e_NB = e_NB.where(lai.gt(3), 0.98).rename('e_NB');
    
    
    e_NB = e_NB.where(ndvi.lt(0).And(image.select('ALFA').lt(0.47)), 0.99).rename('e_NB');
    e_NB = e_NB.where(ndvi.lt(0).And(image.select('ALFA').gte(0.47)), 0.99).rename('e_NB');
    
    log_eNB = e_NB.log();   
    #LLand Surface Temperature
    comp_onda = ee.Number(1.115e-05);        
    lst = image.expression(
      'Tb / ( 1+ ( ( comp_onda * Tb / fator) * log_eNB))',{
        'Tb': image.select('BRT').divide(10),
        'comp_onda': comp_onda,
        'log_eNB': log_eNB,
        'fator': ee.Number(1.438e-02),
      }).rename('T_LST');
    
    
    return image.addBands([lst]) 


































