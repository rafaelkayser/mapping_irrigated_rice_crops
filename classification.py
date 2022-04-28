# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:06:00 2022

@author: rafae
"""


import ee



from mask import fc_masking_global_datasets, fc_masking_mapbiomas



def fc_random_forest(year, roi, img_clas_base, samples_training, input_scale):
    
   
    #---------------------------------------------------------------------------------------
    #INSERT NO CROP DATABASES
    nocrops = fc_masking_mapbiomas(year,roi)
    img_clas_base = img_clas_base.updateMask(nocrops.eq(0))
    
    
    #// REPROJECT TO 60 METERS
    img_clas_base = img_clas_base.reproject('EPSG:32721', None, input_scale);

    
    training = img_clas_base.sampleRegions(
             collection= samples_training,
             properties= ["class"],
             tileScale= 16,
             scale= input_scale);
             
             
    classifier = ee.Classifier.smileRandomForest(300).train(
             features= training,
             classProperty= "class");

    #dict_clas = classifier.explain();
    #importance = dict_clas.get('importance')
    #feat_importance = ee.FeatureCollection( ee.Feature(None, importance));
    
 
    #// Classify the composite.
    img_clas_output = img_clas_base.classify(classifier);        
    img_clas_output = img_clas_output.rename('classification');


    
    return img_clas_output
    
    


def fc_cluster_clas_original(year, roi, img_clas_base, input_scale, band_cluster, nclusters):
    
                   

    #if (flag_mask==1):   
    nocrops = fc_masking_mapbiomas(year,roi)
    img_clas_base = img_clas_base.updateMask(nocrops.eq(0))
    
    
    #// REPROJECT TO 60 METERS
    img_clas_base = img_clas_base.reproject('EPSG:32721', None, input_scale);
    
    
    #// filtro: kernel (dist. gaussiana)
    gaussKernel = ee.Kernel.gaussian(
            radius = 2,
            sigma = 1,
            units = 'pixels',
            normalize = True,
            magnitude = 1);


    img_clas_base = img_clas_base.convolve(gaussKernel);

    
    #-------------------------------------------------------------------------------------------
    #// Make the training dataset.
    training = img_clas_base.sample(
                region = roi.geometry(),
                scale= input_scale,
                numPixels= 20000,
                tileScale= 16)
         
    #// Instantiate the clusterer and train it.
    clusterer = ee.Clusterer.wekaKMeans(nclusters).train(training);
     
      
    img_classified_raw = img_clas_base.cluster(clusterer);
         
         
         
         
         
    #///////////////////////////////////////////////////////
    #atribuir classificação

    #/// Define groups using the training point set: 
    clusteredgroup = training.cluster(clusterer, "cluster")

    groups = clusteredgroup.reduceColumns(ee.Reducer.mean().group(0), ["cluster", band_cluster])


    #/// Extract the cluster IDs and the means.
    groups = ee.List(groups.get("groups"))
         
         
    def dict_group(d):
             return ee.Dictionary(d).get('group')
    ids = groups.map(dict_group)
         
    def dict_mean(d):
             return ee.Dictionary(d).get('mean')
    means = groups.map(dict_mean)
        
    
    #/// Sort the IDs using the means as keys: from smallest to largest NDMI
    sortedsentinel = ee.Array(ids).sort(means).toList()


    
    if (nclusters ==3):
        sortedsentinel2 = ee.Array([0,1,2]).toList()
    else:
        sortedsentinel2 = ee.Array([0,1]).toList()        


    #/// Remap the clustered image to put the clusters in order.
    #ordered = img_classified_raw.remap(sortedsentinel, ids)

    #//Map.addLayer(ordered, {min:0, max:1, 'palette':waterPalette}, "Sorted clusters")
    def reordersentinel(image):
        sortedimage = image.remap(sortedsentinel, ids) 
        return sortedimage.remap(sortedsentinel2, ids).rename('crop')
     
    ordered2 = reordersentinel(img_classified_raw)
    img_clas_output = ordered2.rename('classification');


    #img_clas_output = img_clas_output.set('scenario', ind_scenario);
    img_clas_output = img_clas_output.set('year', year);
    
    
    
    if (nclusters ==3):
        img_clas_output = img_clas_output.where(img_clas_output.eq(0), 0)
        img_clas_output = img_clas_output.where(img_clas_output.eq(1), 3)  #rainfed (atribui provisoriamente no 1)
        img_clas_output = img_clas_output.where(img_clas_output.eq(2), 1)
        img_clas_output = img_clas_output.where(img_clas_output.eq(3), 2)
           
    
    return img_clas_output





def fc_cluster_clas_with_mask(year, roi, img_clas_base, input_scale, band_cluster, img_class_crop):
    
    
    nclusters =2
    
                  
    #mask pre-definied no crop image
    img_clas_base = img_clas_base.updateMask(img_class_crop);
        
        
    #if (flag_mask==1):   
    nocrops = fc_masking_mapbiomas(year,roi)
    img_clas_base = img_clas_base.updateMask(nocrops.eq(0))
    
    
    #// REPROJECT TO 60 METERS
    img_clas_base = img_clas_base.reproject('EPSG:32721', None, input_scale);
    
    
    #// filtro: kernel (dist. gaussiana)
    gaussKernel = ee.Kernel.gaussian(
            radius = 2,
            sigma = 1,
            units = 'pixels',
            normalize = True,
            magnitude = 1);


    img_clas_base = img_clas_base.convolve(gaussKernel);

    
    #-------------------------------------------------------------------------------------------
    #// Make the training dataset.
    training = img_clas_base.sample(
                region = roi.geometry(),
                scale= input_scale,
                numPixels= 20000,
                tileScale= 16)
         
    #// Instantiate the clusterer and train it.
    clusterer = ee.Clusterer.wekaKMeans(nclusters).train(training);
    
    img_classified_raw = img_clas_base.cluster(clusterer);
         
         
         
         
         
    #///////////////////////////////////////////////////////
    #atribuir classificação

    #/// Define groups using the training point set: 
    clusteredgroup = training.cluster(clusterer, "cluster")

    groups = clusteredgroup.reduceColumns(ee.Reducer.mean().group(0), ["cluster", band_cluster])


    #/// Extract the cluster IDs and the means.
    groups = ee.List(groups.get("groups"))
         
         
    def dict_group(d):
             return ee.Dictionary(d).get('group')
    ids = groups.map(dict_group)
         
    def dict_mean(d):
             return ee.Dictionary(d).get('mean')
    means = groups.map(dict_mean)
        
    
    #/// Sort the IDs using the means as keys: from smallest to largest NDMI
    sortedsentinel = ee.Array(ids).sort(means).toList()



    sortedsentinel2 = ee.Array([0,1]).toList()        


    #/// Remap the clustered image to put the clusters in order.
    #ordered = img_classified_raw.remap(sortedsentinel, ids)


    def reordersentinel(image):
        sortedimage = image.remap(sortedsentinel, ids) 
        return sortedimage.remap(sortedsentinel2, ids).rename('crop')
     
    ordered2 = reordersentinel(img_classified_raw)
    img_clas_output = ordered2.rename('classification');


    #img_clas_output = img_clas_output.set('scenario', ind_scenario);
    img_clas_output = img_clas_output.set('year', year);
    
    
    #if (type_cluster ==3):    
    # img_clas_output = img_clas_output.where(img_clas_output.eq(1), 2) #hand
    # img_clas_output = img_clas_output.where(img_clas_output.eq(0), 1)
        
        
    img_clas_output = img_clas_output.where(img_clas_output.eq(0), 2)
        
    
    return img_clas_output
    
    
    
    
def fc_majority_filter(img_class_rf_raw):



     img_clas_final = img_class_rf_raw;

     count_pixel = 120;

     img_clas_final = img_clas_final.unmask(0);


     i_clas_irrig = img_clas_final.updateMask(img_clas_final.eq(1));
     i_clas_rain = img_clas_final.updateMask(img_clas_final.eq(2));

     patchsize1 = i_clas_irrig.connectedPixelCount(count_pixel, False).reproject(ee.Image(img_clas_final).projection());
     patchsize2 = i_clas_rain.connectedPixelCount(count_pixel, False).reproject(ee.Image(img_clas_final).projection());
     
     
     
     #// run a majority filter
     filtered = img_clas_final.focal_mode(
         radius= 200,
         kernelType= 'circle',
         units= 'meters',
      ); 


     #// SUBSTITUI FOCAL MODE NOS PIXELS ISOLADOS
     lim_subs = count_pixel  #// limite de pixels conectados a serem substituidos 

     img_clas_final_filt =  img_clas_final.where(patchsize1.lt(lim_subs),filtered);
     img_clas_final_filt =  img_clas_final_filt.where(patchsize2.lt(lim_subs),filtered);

     img_clas_final_filt = img_clas_final_filt.updateMask(img_clas_final_filt.gt(0));

     #// EXCLUI PIXELS ISOLADOS

     i_clas_irrig = img_clas_final_filt.updateMask(img_clas_final_filt.eq(1));
     i_clas_rain = img_clas_final_filt.updateMask(img_clas_final_filt.eq(2));

     i_clas_irrig = i_clas_irrig.connectedPixelCount(count_pixel, False).reproject(ee.Image(img_clas_final).projection());
     i_clas_rain = i_clas_rain.connectedPixelCount(count_pixel, False).reproject(ee.Image(img_clas_final).projection());

     #//MASCARA FINAL
     i_clas_irrig =  (i_clas_irrig.updateMask(i_clas_irrig.gte(lim_subs)));
     i_clas_irrig = i_clas_irrig.where(i_clas_irrig.gte(lim_subs), 1).unmask(0);

     i_clas_rain =  (i_clas_rain.updateMask(i_clas_rain.gte(lim_subs)));
     i_clas_rain = i_clas_rain.where(i_clas_rain.gte(lim_subs), 2).unmask(0);


     #//////////////////////////////////////////////////////////////////////////////////////////////////////////////

     img_clas_mf = i_clas_irrig.add(i_clas_rain);

     img_clas_mf = img_clas_mf.updateMask(img_clas_mf.gt(0));
     #img_clas_mf = img_clas_mf.set('scenario', ind_scenario);
     
     return img_clas_mf
 





def fc_accuracy(img_map_rf_final, img_clas_base, roi, input_scale, samples_val, name_band_obs, name_band_calc):
    
    
    col_map_rf_final = img_map_rf_final  # ee.ImageCollection(img_map_rf_final)
    

    #samples_val = img_clas_base.stratifiedSample(
     #       numPoints= 1000, 
     #       classBand= 'class', 
     #       region= roi, 
     #       scale= 60,
     #       tileScale= 4
     #       );
    
    # function to create table with extracted information 
    
    def function_loop_accuracy(image):
    #if (1==1):
        
       # image = img_map_rf_final      
       # image = img_map_rf_final.filterMetadata('scenario', 'equals', 6).first();
        
        image = image.unmask(0)      
        image = image.clip(roi)
        
        #// Extract results for test vectors and generate validation accuracy
        fc_ext_samples_pred = image.sampleRegions(samples_val, None);
          
        confMatrix = fc_ext_samples_pred.errorMatrix(name_band_obs, name_band_calc); #Confusion Matrix

        OA = confMatrix.accuracy()           #Overall Accuracy
        CA = confMatrix.consumersAccuracy()  #Consumers Accuracy
        Kappa = confMatrix.kappa() #
        Order = confMatrix.order()
        PA = confMatrix.producersAccuracy()  #Producers Accuracy
        
        
        scenario = image.get('scenario')
        
        #PAi = PA[1]
        
        #print(PAi.getInfo())
        
        
      #  print(confMatrix.getInfo())
        
        #print(OA.getInfo())
        
        
 
        #validation = test.classify(rfclassifier);
        #errormatrix = validation.errorMatrix('lulc', 'classification')
        #Accuracy = errormatrix.accuracy();
        #prodAccuracy = errormatrix.producersAccuracy(); //(correct / total) for each column
        #userAccuracy = errormatrix.consumersAccuracy(); //(correct / total) for each row
        #Kappa = errormatrix.kappa();


        #//SET EXPORT PARAMS
        table_accuracy = ee.Feature(None,{
                'Scenario': scenario,
                'Error Matrix': confMatrix,
                'Overall Accuracy': OA,
                'Producer Accuracy': PA,
                'User Accuracy': CA,
                'Kappa': Kappa,
                'Order': Order,
        });
    
    
     
    
        return table_accuracy
        
    
    table_accuracy = col_map_rf_final.map(function_loop_accuracy);
    
    
    return table_accuracy

























    