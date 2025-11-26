# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 12:59:24 2022

@author: M273075
"""

from matplotlib import pyplot as plt
#from PIL import Image
import tensorflow as tf
#import json
#import os
from  scipy.ndimage import gaussian_filter
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"        
import sys
import json
import time
import numpy as np
dll_path = 'C:\\Users\\M300305\\Desktop\\openslide-win64-20230414\\bin'
os.add_dll_directory(dll_path)
path=dll_path+os.pathsep+os.environ['PATH']
os.environ['PATH']=path
import time
import matplotlib.pylab as plt
import openslide as osl
# arg0 = sys.argv[0]
# sys.path.append(os.path.join(os.getcwd(),'models'))
# print(arg0)
import matplotlib.cm as cm
from tensorflow import keras
import uuid
import cv2 

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
with open(("configPR_SQC_IY.json")) as json_file:  
   paramsObj = json.load(json_file)


# from ImageDataAugmentation import ImageDataGenerator as dataGenerator
exec('from '+paramsObj['modelPy']+' import prepModel as getModel', globals(),globals())
# exec('from modelTF import prepModel as getModel', globals(),globals())
 

model = getModel(
    trainFlag='thyroidAI',
    root_dir=paramsObj['root_dir'],
    task=paramsObj['task'],
    image_path = paramsObj['image_path'],
    val_path = paramsObj['image_path'],
    label_path = paramsObj['label_path'],
    batch_size = paramsObj['batch_size'], 
    n_classes=paramsObj['n_classes'],
    n_channel=paramsObj['n_channel'],
    image_shape=paramsObj['image_shape'],
    label_shape=paramsObj['label_shape'],
    image_format=paramsObj['image_format'],
    label_format=paramsObj['label_format'],
    model=paramsObj['model'],
    optimizer=paramsObj['optimizer'],
    datagen='',
    afold='',
    project_folder=paramsObj['project_folder'],
    logs_folder=paramsObj['logs_folder'],
    weights_folder=paramsObj['weights_folder'],
    preds_folder=paramsObj['preds_folder'],
    weight_file_name=paramsObj['weight_file_name'],
    output_label=paramsObj['time_stamp'],
    time_stamp=paramsObj['time_stamp'],
    init_model =paramsObj['init_model'],    
    n_epochs=paramsObj['n_epochs'],
    shuffle=paramsObj['shuffle'],
    loss=paramsObj['loss'],
    weight_loss=paramsObj['weight_loss'],
    class_weights=paramsObj['class_weights'],
    lr=paramsObj['lr'],
    lr_decay=paramsObj['lr_decay'] )

rmodel=model.create2DModel() 

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)




def dice_coef_multilabel(y_true, y_pred, classes):
    dice=0
    for index in classes:
        dice += dice_coef(y_true[:,:,index], y_pred[:,:,index])
    return dice/len(classes) # taking average

alldice0=[]
alldice1=[]
alldice2=[]
alldice3=[]
alldice4=[]
alldice5=[]
alldice6=[]
alldice7=[]
alldice8=[]
alldice9=[]
alldice10=[]
alldice11=[]
alldice12=[]
alldice13=[]
alldice14=[]
alldice15=[]
alldice16=[]
alldice17=[]


allpreds=[]
alllabels=[]
#countsGT={'cortex':0, "glomerulus":0,"glomerulus-sclerotic":0,"lumen":0,"tubule":0}
#countsPD={'cortex':0, "glomerulus":0,"glomerulus-sclerotic":0,"lumen":0,"tubule":0}
countsGT={'cortex':0, "glomerulus":0,"glomerulus-sclerotic":0,"lumen":0,"tubule":0, 'background':0, "medulla":0, "capsule":0,"other-tissue":0, "artery":0,"tubule-atrophy":0, "interstitial-tissue":0,"interstitial-fibrosis":0,"arteriole":0,"tubule-proximal":0,"tubule-distal":0,"artifact":0,"other":0}
countsPD={'cortex':0, "glomerulus":0,"glomerulus-sclerotic":0,"lumen":0,"tubule":0, 'background':0, "medulla":0, "capsule":0,"other-tissue":0, "artery":0,"tubule-atrophy":0, "interstitial-tissue":0,"interstitial-fibrosis":0,"arteriole":0,"tubule-proximal":0,"tubule-distal":0,"artifact":0,"other":0}
#directory='C:\\Users\\m300305\\Desktop\\mayo\\kidney\\kidney_right\\new_level1'
directory='C:\\Users\\m300305\\Desktop\\kidney\\kidney_dataset\\val'
pcounter=0
for filename in os.listdir(directory):
    filepath=os.path.join(directory,filename)
    dataset=tf.data.TFRecordDataset(filepath)

    n_classes=18
    num_of_images = sum(1 for _ in tf.data.TFRecordDataset(filepath))
    print('Total num of images=',num_of_images)
    def decode(serialized_example):
            """
            Parses an image and label from the given `serialized_example`.
            It is used as a map function for `dataset.map`
            """

            # 1. define a parser
            features = tf.io.parse_single_example(
                serialized_example,
                # Defaults are not specified since both keys are required.
                features = {
                    'height': tf.io.FixedLenFeature([], tf.int64),
                    'width': tf.io.FixedLenFeature([], tf.int64),
                    'zlevel': tf.io.FixedLenFeature([], tf.int64),
                    'oh': tf.io.FixedLenFeature([], tf.int64),
                    'ow': tf.io.FixedLenFeature([], tf.int64),
                    'image_raw': tf.io.FixedLenFeature([], tf.string),
                    'mask_raw': tf.io.FixedLenFeature([], tf.string),
                    'pred_label':  tf.io.FixedLenFeature([], tf.int64),
                    'prj_name': tf.io.FixedLenFeature([], tf.string),
                    'img_name': tf.io.FixedLenFeature([], tf.string),
                    'label_text':  tf.io.FixedLenFeature([], tf.string),
                    'label_set':  tf.io.FixedLenFeature([], tf.string),
                    'cpath':  tf.io.FixedLenFeature([], tf.string),
                    'cfolder':  tf.io.FixedLenFeature([], tf.string),
                    'cfile':  tf.io.FixedLenFeature([], tf.string),
                })


            # 2. Convert the data
            labels = tf.cast(features['pred_label'], tf.int32)
            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            images = tf.io.decode_raw(features['image_raw'], tf.uint8)
            masks = tf.io.decode_raw(features['mask_raw'], tf.uint8)
            images=tf.reshape(images,[height,width,3])  
            masks = tf.reshape(masks,[height,width,1]) 
            
            #print(labels)
            labels = tf.cast(tf.reshape(labels,[1]),tf.uint8)
            labels = tf.one_hot(labels, n_classes)
            labels = tf.reshape(labels, [n_classes])
            #print(labels.shape)

            return images, masks ,labels    

    
    llist=['background','cortex', "medulla", "capsule","other-tissue", "lumen", "glomerulus","glomerulus-sclerotic","artery","tubule","tubule-atrophy", "interstitial-tissue","interstitial-fibrosis","arteriole","tubule-proximal","tubule-distal","artifact","other"]

    lCounter={}
    lset={}
    labelset={}
    for i in range(len(llist)):
        lCounter[llist[i]]=0
        labelset[llist[i]]=i
        lset[i]=llist[i]

    data=dataset.map(decode)
    label = []
    ts=1024
    pcounter+=num_of_images
    for i in data.take(num_of_images):
        arr = i[2].numpy()
        curr_lab=int(np.argmax(arr))
        #print(curr_lab)

        lCounter[llist[curr_lab]]+=1

        image=i[0].numpy()
        mask_gt=i[1].numpy()
        label_gt=np.argmax(i[2].numpy())
        # newimg=Image.fromarray(image[512:,512:,:],'RGB')
        polys,label_pred, mask_pred=model.run(rmodel,np.asarray(image))

        mask_gt=tf.one_hot(np.squeeze(mask_gt), n_classes).numpy()
        mask_pred=tf.one_hot(mask_pred, n_classes).numpy()
        mask_gt=mask_gt.astype('uint8')
        mask_pred=mask_pred.astype('uint8')

        #classes=[1]
        #alldice1.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        #classes=[6]
        #alldice6.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        #classes=[7]
        #alldice7.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        #classes=[5]
        #alldice5.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        #classes=[9]
        #alldice9.append(dice_coef_multilabel(mask_gt,mask_pred,classes))

        classes=[0]
        alldice0.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[1]
        alldice1.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[2]
        alldice2.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[3]
        alldice3.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[4]
        alldice4.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[5]
        alldice5.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[6]
        alldice6.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[7]
        alldice7.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[8]
        alldice8.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[9]
        alldice9.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[10]
        alldice10.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[11]
        alldice11.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[12]
        alldice12.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[13]
        alldice13.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[14]
        alldice14.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[15]
        alldice15.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[16]
        alldice16.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        classes=[17]
        alldice17.append(dice_coef_multilabel(mask_gt,mask_pred,classes))
        

        
        
        fig, ax = plt.subplots(1, 3, figsize = (8, 8))
        im = ax[0].imshow(np.argmax(mask_gt,axis=-1),cmap='jet')
        plt.colorbar(im, ax = ax[0], fraction = 0.046, pad = 0.04)
        ax[0].axis('off')
        ax[0].set_title('Grount-truth')

        im = ax[1].imshow(np.argmax(mask_pred,axis=-1),cmap='jet')
        plt.colorbar(im, ax = ax[1], fraction = 0.046, pad = 0.04)
        ax[1].axis('off')
        ax[1].set_title('Model Prediction')
        
        im = ax[2].imshow(image.astype('uint8'))
        plt.colorbar(im, ax = ax[2], fraction = 0.046, pad = 0.04)
        ax[2].axis('off')
        ax[2].set_title('Original Image')
        plt.show()
        t1=time.time()

        classes=[6,7,9]
        #classes=[6,7,8,9,10,11,12,13,14,15,16,17]
        connectivity=4
        for nc in classes:
            ret1, thresh1 = cv2.threshold(mask_gt[:,:,nc],0,1,cv2.THRESH_BINARY)
            ret2, thresh2 = cv2.threshold(mask_pred[:,:,nc],0,1,cv2.THRESH_BINARY)

            output = cv2.connectedComponentsWithStats(thresh1, connectivity, cv2.CV_32S)
            (numLabels, labelsGT, statsGT, centroids) = output
            idxGT=np.where(statsGT[:,4]<500)
            for m in range(len(idxGT[0])):
                idx=np.where(labelsGT==idxGT[0][m])
                labelsGT[idx]=0
            countsGT[lset[nc]]=countsGT[lset[nc]]+len(np.unique(labelsGT))
            output = cv2.connectedComponentsWithStats(thresh2.astype('uint8'), connectivity, cv2.CV_32S)
            (numLabels, labelsPD, statsPD, centroids) = output
            if nc==6 or nc==7:
                idxPD=np.where(statsPD[:,4]<5000)
            else:
                idxPD=np.where(statsPD[:,4]<500)
                
            for m in range(len(idxPD[0])):
                idx=np.where(labelsPD==idxPD[0][m])
                labelsPD[idx]=0
            countsPD[lset[nc]]=countsPD[lset[nc]]+len(np.unique(labelsPD))
t=1
        # fig, ax = plt.subplots(1, 2, figsize = (8, 8))
        
        # im = ax[0].imshow(mask_gt)
        # plt.colorbar(im, ax = ax[0], fraction = 0.046, pad = 0.04)
        # ax[0].axis('off')
        # ax[0].set_title('ground truth')

        # im = ax[1].imshow(mask_pred, 'jet')
        # plt.colorbar(im, ax = ax[1], fraction = 0.046, pad = 0.04)
        # ax[1].axis('off')
        # ax[1].set_title('prediction')

        # plt.show()

        
print(lCounter)
#print("Dice Score for Class 1 (cortex):", np.mean(alldice1))
#print("Dice Score for Class 6 (glomerulus):", np.mean(alldice6))
#print("Dice Score for Class 7 (glomerulus-sclerotic):", np.mean(alldice7))
#print("Dice Score for Class 5 (lumen):", np.mean(alldice5))
#print("Dice Score for Class 9 (tubule):", np.mean(alldice9))

#print("Dice Score for Class 1 (cortex):", np.mean(alldice1))
#print("Dice Score for Class 6 (glomerulus):", np.mean(alldice6))
#print("Dice Score for Class 7 (glomerulus-sclerotic):", np.mean(alldice7))
#print("Dice Score for Class 5 (lumen):", np.mean(alldice5))
#print("Dice Score for Class 9 (tubule):", np.mean(alldice9))



print("Dice Score for Class 0 (background):", np.mean(alldice0))
print("Dice Score for Class 1 (cortex):", np.mean(alldice1))
print("Dice Score for Class 2 (medulla):", np.mean(alldice2))
print("Dice Score for Class 4 (capsule):", np.mean(alldice3))
print("Dice Score for Class 4 (other-tissue):", np.mean(alldice4))
print("Dice Score for Class 5 (lumen):", np.mean(alldice5))
print("Dice Score for Class 6 (glomerulus):", np.mean(alldice6))
print("Dice Score for Class 7 (glomerulus-sclerotic):", np.mean(alldice7))
print("Dice Score for Class 8 (artery):", np.mean(alldice8))
print("Dice Score for Class 9 (tubule):", np.mean(alldice9))
print("Dice Score for Class 10 (tubule-atrophy):", np.mean(alldice10))
print("Dice Score for Class 11 (interstitial-tissue):", np.mean(alldice11))
print("Dice Score for Class 12 (interstitial-fibrosis):", np.mean(alldice12))
print("Dice Score for Class 13 (arteriole):", np.mean(alldice13))
print("Dice Score for Class 14 (tubule-proximal):", np.mean(alldice14))
print("Dice Score for Class 15 (tubule-distal):", np.mean(alldice15))
print("Dice Score for Class 16 (artifact):", np.mean(alldice16))
print("Dice Score for Class 17 (other):", np.mean(alldice17))

# Calculate and print the total Dice score
#total_dice_weighted = (
    #(np.mean(alldice1) * countsGT['cortex']) +
    #(np.mean(alldice6) * countsGT['glomerulus']) +
    #(np.mean(alldice7) * countsGT['glomerulus-sclerotic']) +
    #(np.mean(alldice5) * countsGT['lumen']) +
    #(np.mean(alldice9) * countsGT['tubule'])
#) / #(countsGT['cortex'] + countsGT['glomerulus'] + countsGT['glomerulus-sclerotic'] + countsGT['lumen'] + countsGT['tubule'])

total_dice_weighted = (
    (np.mean(alldice0) * countsGT['background']) +
    (np.mean(alldice1) * countsGT['cortex']) +
    (np.mean(alldice2) * countsGT['medulla']) +
    (np.mean(alldice3) * countsGT['capsule']) +
    (np.mean(alldice4) * countsGT['other-tissue']) +
    (np.mean(alldice5) * countsGT['lumen']) +
    (np.mean(alldice6) * countsGT['glomerulus']) +
    (np.mean(alldice7) * countsGT['glomerulus-sclerotic']) +
    (np.mean(alldice8) * countsGT['artery']) +
    (np.mean(alldice9) * countsGT['tubule']) +
    (np.mean(alldice10) * countsGT['tubule-atrophy']) +
    (np.mean(alldice11) * countsGT['interstitial-tissue']) +
    (np.mean(alldice12) * countsGT['interstitial-fibrosis']) +
    (np.mean(alldice13) * countsGT['arteriole']) +
    (np.mean(alldice14) * countsGT['tubule-proximal']) +
    (np.mean(alldice15) * countsGT['tubule-distal']) +
    (np.mean(alldice16) * countsGT['artifact']) +
    (np.mean(alldice17) * countsGT['other'])
) / (
    countsGT['background'] + countsGT['cortex'] + countsGT['medulla'] +
    countsGT['capsule'] + countsGT['other-tissue'] + countsGT['lumen'] +
    countsGT['glomerulus'] + countsGT['glomerulus-sclerotic'] + countsGT['artery'] +
    countsGT['tubule'] + countsGT['tubule-atrophy'] + countsGT['interstitial-tissue'] +
    countsGT['interstitial-fibrosis'] + countsGT['arteriole'] + countsGT['tubule-proximal'] +
    countsGT['tubule-distal'] + countsGT['artifact'] + countsGT['other']
)

print("Weighted Total Dice Score:", total_dice_weighted)







