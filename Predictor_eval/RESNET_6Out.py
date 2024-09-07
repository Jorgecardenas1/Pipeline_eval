"""Version 2: RESNET Eval for training version 5"""

import sys
import os
import time
#from Utilities.SaveAnimation import Video
import uuid
from druida import Stack
from druida import setup
from druida.DataManager import datamanager
from druida.tools import utils

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optimizer

from torchsummary import summary
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy
from torchvision.utils import save_image

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda
import torchvision.utils as vutils
from torch.autograd import Variable
from torcheval.metrics.functional import r2_score
from scipy.signal import find_peaks,peak_widths

import glob
from tqdm.notebook import tqdm
import argparse
import json
from PIL import Image
import cv2 

# CAD

from druida.tools.utils import CAD 


# Arguments
parser = argparse.ArgumentParser()
# boxImagesPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Images Jorge Cardenas 512\\"
# DataPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Exports\\output\\"
# simulationData="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\DBfiles\\"

boxImagesPath="../../../data/MetasufacesData/Images-512-Bands/"
#boxImagesPath="../../../data/MetasufacesData/Images-512-Suband/"

DataPath="../../../data/MetasufacesData/Exports/output/"
simulationData="../../../data/MetasufacesData/DBfiles/"
validationImages="../../../data/MetasufacesData/testImages/"

Substrates={"Rogers RT/duroid 5880 (tm)":0, "other":1}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
#TargetGeometries={"circ":0,"box":1, "cross":2}
TargetGeometries={"circ":[1,0,0],"box":[0,1,0], "cross":[0,0,1]}
Bands={"30-40":0,"40-50":1, "50-60":2,"60-70":3,"70-80":4, "80-90":5}

#Bands={"30-40":[1,0,0,0,0,0],"40-50":[0,1,0,0,0,0], 
#       "50-60":[0,0,1,0,0,0],"60-70":[0,0,0,1,0,0],"70-80":[0,0,0,0,1,0], 
#       "80-90":[1,0,0,0,0,1]}
Height={"0.252":[1,0,0,0],"0.508":[0,1,0,0], 
       "0.787":[0,0,1,0],"1.575":[0,0,0,1]}

def arguments():


    parser.add_argument("-run_name",type=str)
    parser.add_argument("-epochs",type=int)
    parser.add_argument("-batch_size",type=int)
    parser.add_argument("-workers",type=int)
    parser.add_argument("-gpu_number",type=int)
    parser.add_argument("-dataset_path",type=str)
    parser.add_argument("-image_size",type=int)
    parser.add_argument("-device",type=str)
    parser.add_argument("-learning_rate",type=float)
    parser.add_argument("-condition_len",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-metricType",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-cond_channel",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-resnet_arch",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-model",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-one_hot_encoding",type=int)
    parser.add_argument("-output_path",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-working_path",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-pred_size",type=int) #This defines the length of our conditioning vector

    parser.run_name = args["-run_name"]
    parser.epochs =  args["-epochs"]
    parser.batch_size = args["-batch_size"]
    parser.workers=args["-workers"]
    parser.gpu_number=args["-gpu_number"]
    parser.image_size = args["-image_size"]
    parser.dataset_path = args["-dataset_path"]
    parser.device = args["-device"]
    parser.learning_rate = args["-learning_rate"]
    parser.condition_len = args["-condition_len"] #Incliuding 3 top frequencies
    parser.metricType= args["-metricType"] #this is to be modified when training for different metrics.
    parser.cond_channel= args["-cond_channel"] #this is to be modified when training for different metrics.
    parser.resnet_arch= args["-resnet_arch"] #this is to be modified when training for different metrics.
    parser.model= args["-model"] #this is to be modified when training for different metrics.
    parser.one_hot_encoding = args["-one_hot_encoding"] #if used OHE Incliuding 3 top frequencies
    parser.output_path = args["-output_path"] #if used OHE Incliuding 3 top frequencies
    parser.working_path = args["-working_path"] #if used OHE Incliuding 3 top frequencies
    parser.pred_size=args["-pred_size"]#this is to be modified when training for different metrics.





# Data pre-processing
def join_simulationData():
    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    
    df.to_csv('out.csv',index=False)
    

# Load Model

def get_net_resnet(device,hiden_num=1000,dropout=0.1,features=3000, Y_prediction_size=601):
    model = Stack.Predictor_RESNET(parser.resnet_arch,conditional=True, cond_input_size=parser.condition_len, 
                                   cond_channels=parser.cond_channel, 
                                ngpu=1, image_size=parser.image_size ,
                                output_size=8, channels=3,
                                features_num=features,hiden_num=hiden_num, #Its working with hiden nums. Features in case and extra linear layer
                                dropout=dropout, 
                                Y_prediction_size=Y_prediction_size) #size of the output vector in this case frenquency points

    model.load_state_dict(torch.load(parser.model))
    model.eval()
    model.cuda()
    criterion=nn.MSELoss()

    return model, criterion 



def prepare_data(names, device,df,classes,classes_types):

    bands_batch=[]
    max_freqs = []
    a = []        
    conditional_tensor=[]

    """Reading Data"""
    for idx,name in enumerate(names):

        series=name.split('_')[-2]#
        band_name=name.split('_')[-1].split('.')[0]#
        batch=name.split('_')[4]
        version_batch=1
        if batch=="v2":
            version_batch=2
            batch=name.split('_')[5]

        for file_name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
            #loading the absorption data
            train = pd.read_csv(file_name)


            # # the band is divided in chunks 
            if Bands[str(band_name)]==0:
            
                train=train.loc[1:100]

            elif Bands[str(band_name)]==1:
                
                train=train.loc[101:200]

            elif Bands[str(band_name)]==2:
                if version_batch==1:
                    train=train.loc[201:300]
                else:
                    train=train.loc[1:100]
            elif Bands[str(band_name)]==3:
                if version_batch==1:
                    train=train.loc[301:400]
                else:
                    train=train.loc[101:200]

            elif Bands[str(band_name)]==4:
                if version_batch==1: 
                    train=train.loc[401:500]
                else:
                    train=train.loc[201:300]

            elif Bands[str(band_name)]==5:

                train=train.loc[501:600]


            """Preparing Label and conditioning"""
            
            #preparing data from spectra for each image
            data=np.array(train.values.T)
            values=data[1]
            all_frequencies=data[0]
            all_frequencies = np.array([(float(i)-min(all_frequencies))/(max(all_frequencies)-min(all_frequencies)) for i in all_frequencies])

            values[values < 0.05]= 0

            #get top freqencies for top values 
            data, fre_peaks,results_half, peaks = find_peaks_values(values,all_frequencies)
            
            #creating the labels tensor with top values and its frequencies
            labels_tensor=torch.cat((torch.from_numpy(data),torch.from_numpy(fre_peaks)),0)
            #labels_tensor=torch.from_numpy(values)

            
            #batching
            a.append(labels_tensor)
            #a.append(torch.from_numpy(values[1]))

            #saving bands and frequencies 
            bands_batch.append(Bands[band_name])
            max_freqs.append(all_frequencies[peaks])
                
            

    if parser.one_hot_encoding == 1:  
        pass
        # conditional_data,sustratoHeight = set_conditioning_one_hot(df,
        #                                     name,
        #                                     classes[idx],
        #                                     classes_types[idx],
        #                                     Bands[str(band_name)],
        #                                     top_frequencies,
        #                                     substrate_encoder,
        #                                     materials_encoder,
        #                                     surfaceType_encoder,
        #                                     TargetGeometries_encoder,
        #                                     bands_encoder)

    else:
        conditional_tensor,sustratoHeight = set_conditioning(bands_batch,
                                    None,
                                    classes, 
                                    names, 
                                    classes_types,
                                    None,
                                    df,
                                    device)

    return a,conditional_tensor

def find_peaks_values(values, all_frequencies):
        peaks = find_peaks(values, threshold=0.00001)[0] #indexes of peaks
        results_half = peak_widths(values, peaks, rel_height=0.5) #4 arrays: widths, y position, initial and final x
        results_half = results_half[0]
        data = values[peaks]
        fre_peaks = all_frequencies[peaks]
        length_output=3

        if len(peaks)>length_output:
            data = data[0:length_output]
            fre_peaks = fre_peaks[0:length_output]
            results_half = results_half[0:length_output]

        elif len(peaks)==0:

            data = np.zeros(length_output)
            fre_peaks = all_frequencies[0:length_output]
            results_half = np.zeros(length_output)

        else:

            difference = length_output-len(peaks)

            for idnx in range(difference):
                data = np.append(data, 0)
                fequencies = np.where(values<0.1)
                fequencies = np.squeeze(fequencies)
                fre_peaks = np.append(fre_peaks,all_frequencies[fequencies[idnx]])
                results_half = np.append(results_half,0)

        return data, fre_peaks,results_half, peaks


# Conditioning
def set_conditioning(bands_batch,freqx_val,target,path,categories,clipEmbedder,df,device):
        

    arr=[] #intermediate array to build the batch of conditioning
    values_array=torch.tensor((), dtype=torch.float) # the conditioning tensor

    for idx,name in enumerate(path):

        series=name.split('_')[-2]
        batch=name.split('_')[4]
        if batch=="v2":
            batch=name.split('_')[5]      
        
        iteration=series.split('-')[-1]
        row=df[(df['sim_id']==batch) & (df['iteration']==int(iteration))  ]

        #getting values from design parameters
        target_val=target[idx]
        category=categories[idx]
        geometry=TargetGeometries[category]
        band=bands_batch[idx]
        

        """"
        surface type: reflective, transmissive
        layers: conductor and conductor material / Substrate information
        """
       
        
        surfacekey=row["type"].values[0]
        surfacetype=Surfacetypes[surfacekey]
        
        layers=row["layers"].values[0]
        layers= layers.replace("'", '"')
        layer=json.loads(layers)
        
        materialconductor=Materials[layer['conductor']['material']]
        materialsustrato=Substrates[layer['substrate']['material']]
        
        
        if (target_val==2): #is cross. Because an added variable to the desing 
            sustratoHeight= json.loads(row["paramValues"].values[0])
            sustratoHeight= sustratoHeight[-2]
            sustratoHeight = Height[str(sustratoHeight)]

            substrateWidth = json.loads(row["paramValues"].values[0])[-1] # from the simulation crosses have this additional free param
        else:
        
            sustratoHeight= json.loads(row["paramValues"].values[0])
            sustratoHeight= sustratoHeight[-1]
            sustratoHeight = Height[str(sustratoHeight)]
            substrateWidth = 5 # 5 mm size
        

        """this to apply for one hot encoding"""
        # materialsustrato=torch.Tensor(substrate_encoder.transform(np.array(Substrates[layer['substrate']['material']]).reshape(-1, 1)).toarray()).squeeze(0)
        # materialconductor=torch.Tensor(materials_encoder.transform(np.array(Materials[layer['conductor']['material']]).reshape(-1, 1)).toarray()).squeeze(0)
        # surface=torch.Tensor(surfaceType_encoder.transform(np.array(Surfacetypes[surfacekey]).reshape(-1, 1)).toarray()).squeeze(0)
        # band=torch.Tensor(bands_encoder.transform(np.array(band).reshape(-1, 1)).toarray()).squeeze(0)
  

        #values_array = torch.cat((torch.Tensor(geometry),torch.Tensor(surfacetype),torch.Tensor(materialconductor),torch.Tensor(materialsustrato),torch.Tensor([sustratoHeight]),torch.Tensor(band)),0) #concat side
        #if parser.pred_size==1:
        #    arr.append(torch.Tensor([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight, substrateWidth,band,freqx_val]))
       
        #else:
        val_arr=torch.Tensor(geometry)
        val_arr=torch.cat((val_arr,torch.Tensor([band])),0)
        #val_arr = torch.Tensor([band,freqx_val])
        arr.append(val_arr)

    #embedding = torch.stack(arr)
    values_array = torch.stack(arr)
    """ Values array solo pouede llenarse con n+umero y no con textos"""
    # values_array = torch.Tensor(values_array)

    return values_array,sustratoHeight




def test(model,criterion,device):
    
    df = pd.read_csv("out.csv")
    
    vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1,
                                            validationImages,parser.batch_size, 
                                            drop_last=True,
                                            filter="30-40")
    
    for i, data in enumerate(vdataloader, 0):

        
        inputs, classes, names, classes_types = data
        inputs = inputs.to(device)
        classes = classes.to(device)
        labels,condition= prepare_data(names, device,df,classes,classes_types)

        if condition.shape[1]==parser.condition_len:

            condition = torch.nn.functional.normalize(condition, p=2.0, dim=1, eps=1e-5, out=None)
            print(inputs.shape)
            y_predicted=model(input_=inputs, conditioning=condition.to(device) ,b_size=inputs.shape[0])
            y_predicted=y_predicted.to(device)
            
            #y_predicted = m(y_predicted)
            y_truth = torch.stack(labels).to(device)

            print(classes,classes_types)
            print(y_predicted)
            print(y_truth)

            err = criterion(y_predicted.float(), y_truth.float())  

            print(err)
            

        break

def testwithLabels(netG,device):
    
    #geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,band
    
    labels=np.array([ 1.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.7870,  0.0000, 0.0000,  1.0000,  0.0000,  0.0000,  0.0000, 53.6000, 52.1000, 53.5000,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.8,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    """
         surface,materialconductor,materialsustrato,torch.Tensor([sustratoHeight]),band,top_freqs
    """
    """
    "Reflective","Transmissive"
    "copper","pec"
    Rogers RT/duroid 5880 (tm)","other"
    height
    "30-40","40-50", "50-60","60-70","70-80", "80-90"
    top 3 freqs in such band
    """
    
    latent_tensor=torch.rand(parser.latent)
    labels_tensor=torch.from_numpy(labels)
    tensor1 = torch.cat((labels_tensor.to(device),latent_tensor.to(device),)) #concat side
          
    testTensor = tensor1.type(torch.float).to(device)

    tensor2 = testTensor.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
    tensor3 = tensor2.permute(1,0,2,3)
    

    fake = netG(tensor3).detach().cpu()


    t_np = torch.squeeze(tensor3,(2,3)).cpu().numpy() #convert to Numpy array
    df_ = pd.DataFrame(t_np) #convert to a dataframe

    df_.to_csv("testfile_labels",index=True) #save to file

    # let's resize our image to be 150 pixels wide, but in order to
    # prevent our resized image from being skewed/distorted, we must
    # first calculate the ratio of the *new* width to the *old* width 
    save_image(fake, str("withLabels")+"_"+str("64")+'.png')

    image = cv2.imread(str("withLabels")+"_"+str("64")+'.png')

    r = 512.0 / image.shape[1]
    dim = (512, int(image.shape[0] * r))
    # perform the actual resizing of the image using cv2 resize
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite("Test_512.png", resized) 


def encoders(dictionary):
    index = []
    for x,y in dictionary.items():
        index.append([y])

    index = np.asarray(index)
    enc = OneHotEncoder()
    enc.fit(index)
    return enc



def main():

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    arguments()
    join_simulationData()  

    net, criterion=get_net_resnet(device,hiden_num=1000,dropout=0.4,features=1000, Y_prediction_size=parser.pred_size)
    net = net.to(device)
    print(net)

    test(net,criterion,device)
    #testwithLabels(netG,device)

if __name__ == "__main__":

    name = str(uuid.uuid4())[:8]

    args =  {"-model":"models/trainedModelTM_abs__RESNET152_Bands_5sep_2e-4_20epc_ADAM_6out.pth",
                                       "-run_name":"Predictor ",
                                       "-epochs":1,
                                       "-batch_size":8,
                                       "-workers":1,
                                       "-gpu_number":1,
                                       "-image_size":128,
                                       "-dataset_path": os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/'),
                                       "-device":"cpu",
                                       "-learning_rate":2e-5,
                                       "-metricType":"AbsorbanceTM",
                                       "-cond_channel":3,
                                       "-condition_len":4,
                                       "-resnet_arch":"resnet152",
                                       "-one_hot_encoding":0,
                                       "-pred_size":6,
                                       "-working_path":"./Predictor_eval/",
                                       "-output_path":"../output/"+name} #OJO etepath lo lee el otro main


    main()