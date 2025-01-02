
"""
Version 2 : implementa GAn V1 y GAN V2
Version 3:
"""
import sys
import os
import time
#from Utilities.SaveAnimation import Video
from typing import Sequence, Optional
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
from torchvision.utils import save_image

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda
import torchvision.utils as vutils
from torch.autograd import Variable
from scipy.signal import find_peaks,peak_widths

import glob
from tqdm.notebook import tqdm
import argparse
import json
from PIL import Image
import cv2 

# CAD

from druida.tools.utils import CAD 

print('Check import functions: Done.')

# Arguments
parser = argparse.ArgumentParser()
# boxImagesPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Images Jorge Cardenas 512\\"
# DataPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Exports\\output\\"
# simulationData="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\DBfiles\\"

boxImagesPath="../../../data/MetasurfacesDataV3/Images-512-Bands/"
#boxImagesPath="../../../data/MetasufacesData/Images-512-Suband/"
DataPath="../../../data/MetasurfacesDataV3/Exports/output/"
simulationData="../../../data/MetasurfacesDataV3/DBfiles/"
validationImages="../../../data/MetasurfacesDataV3/testImages/"


Substrates={"Rogers RT/duroid 5880 (tm)":0, "other":1}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
TargetGeometries={"circ":[1,0,0],"box":[0,1,0], "cross":[0,0,1]}
Bands={"75-78":0}



def arguments(args):
    
    parser.add_argument("-run_name",type=str)
    parser.add_argument("-epochs",type=int)
    parser.add_argument("-batch_size",type=int)
    parser.add_argument("-workers",type=int)
    parser.add_argument("-gpu_number",type=int)
    parser.add_argument("-image_size",type=int)
    parser.add_argument("-dataset_path",type=str)
    parser.add_argument("-device",type=str)
    parser.add_argument("-learning_rate",type=float)
    parser.add_argument("-condition_len",type=float) #This defines the length of our conditioning vector
    parser.add_argument("-metricType",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-latent",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-spectra_length",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-gen_model",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-one_hot_encoding",type=int)
    parser.add_argument("-output_path",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-working_path",type=str) #This defines the length of our conditioning vector
    parser.add_argument("GAN_version",type=bool)
    parser.add_argument("output_channels", type=int)

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
    parser.latent=args["-latent"] #this is to be modified when training for different metrics.
    parser.spectra_length=args["-spectra_length"] #this is to be modified when training for different metrics.
    parser.gen_model=args["-gen_model"]
    parser.one_hot_encoding = args["-one_hot_encoding"] #if used OHE Incliuding 3 top frequencies
    parser.output_path = args["-output_path"] #if used OHE Incliuding 3 top frequencies
    parser.working_path = args["-working_path"] #if used OHE Incliuding 3 top frequencies
    parser.GAN_version=True
    parser.output_channels=args["-output_channels"]

    print('Check arguments function: Done.')


#From the DCGAN paper, the authors specify that all model weights shall be randomly initialized
#from a Normal distribution with mean=0, stdev=0.02.
#The weights_init function takes an initialized model as input and reinitializes all convolutional,
#convolutional-transpose, and batch normalization layers to meet this criteria.

# Data pre-processing
def join_simulationData():


    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    
    df.to_csv('out.csv',index=False)

print('Check join_simulationData function: Done.')
    
def cad_generation(images_folder,destination_folder,image_file_name,sustratoHeight,cellsize):

    ImageProcessor=CAD("./"+images_folder+"generated_image_512.png", "./"+images_folder+"processed/")
    image_name="./"+images_folder+image_file_name

    upperBound=[50,255,255]

    #Feature extraction
    lowerBound=[0,80,80]
    epsilon_coeff=0.0500
    threshold_Value=0
    contour_name="RED"
    red_cnts,size=ImageProcessor.colorContour(upperBound, lowerBound,image_name,epsilon_coeff, threshold_Value,contour_name)

    upperBound=[90, 255,255]
    lowerBound=[36, 100, 0]
    epsilon_coeff=0.0500
    threshold_Value=100
    contour_name="GREEN"

    green_cnts,size=ImageProcessor.colorContour(upperBound, lowerBound,image_name,epsilon_coeff, threshold_Value,contour_name)

    upperBound=[255,255,255]
    lowerBound=[20,0,0]
    epsilon_coeff=0.0500
    threshold_Value=0
    contour_name="BLUE"

    blue_cnts,size=ImageProcessor.colorContour(upperBound, lowerBound,image_name,epsilon_coeff, threshold_Value,contour_name)

    """DXF generation"""
    cellsize = round(cellsize*100,1) + 1
    print(cellsize)
    units="um"
    GoalSize=cellsize
    currentSize=size[0] #assumming an squared image same witdth and height
    multiplier=0
    layerscale=0

    if units=="mm":
        multiplier=0.1
        layerscale=1
    elif units=="um":
        multiplier=10
        layerscale=1000
    else:
        pass


    """Here the original layers were given in mm
    as the target units are nm, layers must be rescaled
    """

    layers= {
            "conductor":{
                "thickness":str(0.035*layerscale),
                "material":"pec",
                "color": "red",
                "zpos":str(sustratoHeight*layerscale)
            },
            "dielectric":{
                "thickness":str(sustratoHeight*layerscale),
                "material":"PTFE",
                "color": "green",
                "zpos":str(sustratoHeight*layerscale)

            } ,
            "substrate":{
                "thickness":str(sustratoHeight*layerscale),
                "material":"Rogers RT/duroid 5880 (tm)",
                "color": "blue",
                "zpos":0

            }
        }

    """We can chose the layers to be included in DXF file to be exporte to HFSS"""
   
    selectedLayer = ["conductor","dielectric" ,"substrate"]
    
    ImageProcessor.DXF_build(multiplier, GoalSize,currentSize,red_cnts, blue_cnts,green_cnts, selectedLayer)

    """This section allows to include information previously known from layers in HFSS model 
    to recreate the model """

    #This info is usuarlly included when generating data by using
    # the data generation module

    kwargs={
            "reports":"",
            "simulation_id":"simID",
        "variable_name":"variables",
            "value" : str([]),
            "units" : units,
        }
    
    ImageProcessor.elevation_file(layers,**kwargs)

print('Check cad_generation function: Done.')


def prepare_data(files_name, device,df,classes,classes_types,substrate_encoder, materials_encoder,surfaceType_encoder,TargetGeometries_encoder,bands_encoder):
    bands_batch,array1,array_labels=[],[],[]

    noise = torch.Tensor()

    for idx,name in enumerate(files_name):

        series=name.split('_')[-2]#
        band_name=name.split('_')[-1].split('.')[0]#
        
        batch=name.split('_')[4]
        version_batch=1

        if batch=="v2":
            version_batch=2
            batch=name.split('_')[5]        #print(files_name)

        for file_name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
            #loading the absorption data
            train = pd.read_csv(file_name)

            # # the band is divided in chunks 
            if Bands[str(band_name)]==0:
                
                train=train.loc[1:100]

            elif Bands[str(band_name)]==1:
                
                train=train.loc[101:200]

    
            #preparing data from spectra for each image
            data_raw=np.array(train.values.T)
            values=data_raw[1]
            all_frequencies=data_raw[0]
            all_frequencies = np.array([(float(i)-min(all_frequencies))/(max(all_frequencies)-min(all_frequencies)) for i in all_frequencies])

            #get top freqencies for top values 
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


            #labels_peaks=torch.cat((torch.from_numpy(data),torch.from_numpy(fre_peaks)),0)
            """this has 3 peaks and its frequencies-9 entries"""
            labels_peaks=torch.cat((torch.from_numpy(data),torch.from_numpy(fre_peaks),torch.from_numpy(results_half)),0)

            """This has geometric params 6 entries"""
            conditional_data,sustratoHeight, substrateWidth = set_conditioning(df,name,classes[idx],
                                                classes_types[idx],
                                                Bands[str(band_name)],
                                                None)



            bands_batch.append(band_name)

            #loading data to tensors for discriminator
            tensorA = torch.from_numpy(values) #Just have spectra profile
            labels = torch.cat((conditional_data.to(device),labels_peaks.to(device),tensorA.to(device))) #concat side    
            labels = torch.nn.functional.normalize(labels, p=2.0, dim=0, eps=1e-5, out=None)

            array_labels.append(labels) # to create stack of tensors
            latent_tensor=torch.randn(1,parser.latent)

            if parser.GAN_version:

                noise = torch.cat((noise.to(device),latent_tensor.to(device)))
            else:
        
                pass

    return array1, array_labels, noise,data_raw,sustratoHeight,substrateWidth


def set_conditioning(df,name,target,categories,band_name,top_freqs):
    series=name.split('_')[-2]

    batch=name.split('_')[4]
    if batch=="v2":
        batch=name.split('_')[5]    
    iteration=series.split('-')[-1]
    row=df[(df['sim_id']==batch) & (df['iteration']==int(iteration))  ]

    print(categories)
    print(batch)
        #print(batch)
        #print(iteration)

    target_val=target
    category=categories
    geometry=TargetGeometries[category]
    band=band_name

    """"
    surface type: reflective, transmissive
    layers: conductor and conductor material / Substrate information
    """
    surfacetype=row["type"].values[0]
    surfacetype=Surfacetypes[surfacetype]
        
    layers=row["layers"].values[0]
    layers= layers.replace("'", '"')
    layer=json.loads(layers)
        
    materialconductor=Materials[layer['conductor']['material']]
    materialsustrato=Substrates[layer['substrate']['material']]
        
        
    if (target_val==2): #is cross. Because an added variable to the desing 
        
        sustratoHeight= json.loads(row["paramValues"].values[0])
        sustratoHeight= sustratoHeight[-2]
        substrateWidth = json.loads(row["paramValues"].values[0])[-1] # from the simulation crosses have this additional free param
    else:
    
        sustratoHeight= json.loads(row["paramValues"].values[0])
        sustratoHeight= sustratoHeight[-2]
        substrateWidth = json.loads(row["paramValues"].values[0])[-1] # from the simulation crosses have this additional free param
        

    #values_array=torch.Tensor(geometry)
    #values_array=torch.cat((values_array,torch.Tensor([sustratoHeight ])),0)
    values_array=torch.Tensor([sustratoHeight ])

    """condition with top frequencies"""
    #values_array = torch.cat((values_array,top_freqs),0) #concat side

    """ conditions no top frequencies"""
    values_array = torch.Tensor(values_array)
    #values_array = torch.nn.functional.normalize(values_array, p=2.0, dim=0, eps=1e-5, out=None)

    return values_array,sustratoHeight, substrateWidth

print('Check set_conditioning function: Done.')

def test(netG,device):
    
    df = pd.read_csv("out.csv")
    
    vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1,
                                            validationImages,parser.batch_size, 
                                            drop_last=True,
                                            filter="30-40")
    
    for i, data in enumerate(vdataloader, 0):
        # Genera el batch del espectro, vectores latentes, and propiedades
        # Estamos Agregando al vector unas componentes condicionales
        # y otras de ruido en la parte latente  .

        inputs, classes, names, classes_types = data
        #sending to CUDA
        inputs = inputs.to(device)
        classes = classes.to(device)
        _, labels, noise,real_values,sustratoHeight,substrateWidth = prepare_data(names, device,df,classes,classes_types,
                                                             None,
                                                             None,
                                                             None,
                                                             None,
                                                             None)

        testTensor = noise.type(torch.float).to(device)

        if parser.GAN_version:
            label_conditions = torch.stack(labels).type(torch.float).to(device) #Discrminator Conditioning spectra
            noise = noise.type(torch.float).to(device) #Generator input espectro+ruido
            label_conditions = torch.nn.functional.normalize(label_conditions, p=2.0, dim=1, eps=1e-5, out=None)

            t_np = noise.cpu().numpy() #convert to Numpy array

            fake = netG(label_conditions,testTensor,parser.batch_size).detach().cpu()

        else:

            pass


        ##Eval
        # we have to process image to obtain size.
        cellsize,fake = recoverSize(fake)

        """Saving Data"""

        if not os.path.exists(parser.output_path):
            os.makedirs(parser.output_path)

        with open(parser.output_path+"/usedmodel.txt", "a") as f:
            f.write("original batch:"+str(names))
            f.write("\n")

            f.write("image_size:"+str(parser.image_size))
            f.write("\n")
            
            f.write("condition_len:"+str(parser.condition_len))
            f.write("\n")

            f.write("latent:"+str(parser.latent))
            f.write("\n")

            f.write("spectra_length:"+str(parser.spectra_length))
            f.write("\n")

            f.write("gen_model:"+str(parser.gen_model))
            f.write("\n")
            
            f.write("one_hot_encoding:"+str(parser.one_hot_encoding))
            f.write("\n")

            f.write("cell size:"+str(cellsize)+" "+"ValidationWidth:"+str(substrateWidth))
            f.write("\n")

        df_ = pd.DataFrame(t_np) #convert to a dataframe
        df_real = pd.DataFrame(real_values)
        #save read data
        df_.to_csv(parser.output_path+"/eval_data",index=True) #save to file
        df_real.to_csv(parser.output_path+"/spectra"+str(classes_types),index=True) #save to file

        plt.plot(real_values[0], real_values[1])
         # Saving figure by changing parameter values
        plt.savefig(parser.output_path+"/spectra"+str(classes_types)+".png")

        
        # upscaling
        save_image(fake, parser.output_path+"/generated_image.png")
        image = cv2.imread(parser.output_path+"/generated_image.png")

        r = 512.0 / image.shape[1]
        dim = (512, int(image.shape[0] * r))
        # perform the actual resizing of the image using cv2 resize
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(parser.output_path+"/generated_image_512.png", resized) 

        imagesFolder = parser.output_path+"/"
        destinationFolder = parser.output_path+"/"
        image_name = "generated_image_512.png"
        time.sleep(3)
        cad_generation(imagesFolder,destinationFolder,image_name,sustratoHeight,cellsize.detach().numpy())
        #plot Data

        break

print('Check test function: Done.')

def recoverSize(image):
   
    fringe_width = 8

    #factor = ((value - 4.85) / (5.15 - 4.85)) 


    # Get image dimensions
    #C, H, W = image.shape
    # Create a mask for the fringe
    mask = torch.zeros((512, 512), dtype=torch.bool)

    # Top fringe
    mask[:fringe_width, :] = True
    # Bottom fringe
    mask[-fringe_width:, :] = True
    # Left fringe
    mask[:, :fringe_width] = True
    # Right fringe
    mask[:, -fringe_width:] = True

    # Expand the mask to match the image dimensions (C, H, W)
    mask = mask.unsqueeze(0).expand(3, -1, -1)
   
    #factor = ((value - 4.85) / (5.15 - 4.85)) 
    fringes = torch.round(image[0][2, mask[0]],decimals=2)
    
    normalized_value = (torch.mode(fringes).values + 1) / 2
    
    old_min, old_max = 4.85, 5.15
    size = normalized_value * (old_max - old_min) + old_min
    print("size:",size)
    image[0][:, mask[0]] = torch.tensor([[-1.0],[-1.0],[1]])

    return size, image

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

print('Check testwithLabels function: Done.')

def encoders(dictionary):
    index = []
    for x,y in dictionary.items():
        index.append([y])

    index = np.asarray(index)
    enc = OneHotEncoder()
    enc.fit(index)
    return enc

print('Check encoders function: Done.')

def main(args):

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    arguments(args)
    join_simulationData()  

    trainer = Stack.Trainer(parser)

    use_GANV2=parser.GAN_version

    if use_GANV2:
        # Create models
        """leakyRelu_flag=False use leaky 
        flag=True use Relu"""
        initial_depth = 512
        generator_mapping_size=64

        netG = Stack.Generator_V2(parser.image_size,
                                  trainer.gpu_number,
                                  parser.spectra_length+parser.condition_len,
                                  parser.latent, generator_mapping_size,
                                  initial_depth,
                                  parser.output_channels,
                                  leakyRelu_flag=False)
        
        netG.load_state_dict(torch.load(parser.gen_model) )  
        #netG.load_state_dict(torch.load(parser.gen_model,map_location=torch.device(device)).state_dict())


        #NETGModelTM_abs__GANV2_FWHM_lowswitch_25Ag-lr1-4.pth
        netG.eval()
        netG.cuda()
    else:

        pass


    #netG = Stack.Generator(trainer.gpu_number, input_size, generator_mapping_size, output_channels)
    

    print(netG)
    test(netG,device)
    #testwithLabels(netG,device)


if __name__ == "__main__":

    name = str(uuid.uuid4())[:8]

    #if not os.path.exists("output/"+str(name)):
    #        os.makedirs("output/"+str(name))
            
    args =  {"-gen_model":"models/NETGModelTM_abs__GANV2_29Dic_HighAbs_512Depth_batch32_512_z400_NoWGeometryCond.pth",
                                       "-run_name":"GAN Training",
                                       "-epochs":1,
                                       "-batch_size":1,
                                       "-workers":1,
                                       "-gpu_number":1,
                                       "-image_size":512,
                                       "-dataset_path": os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/'),
                                       "-device":"cpu",
                                       "-learning_rate":5e-5,
                                       "-condition_len":10,
                                       "-metricType":"AbsorbanceTM",
                                       "-latent":400,
                                       "-output_channels":3,
                                       "-spectra_length":100,
                                       "-one_hot_encoding":0,
                                       "-working_path":"./Generator_eval/",
                                       "-output_path":"../output/"+name} #OJO etepath lo lee el otro main

    main(args)

print('Check main function: Done.')