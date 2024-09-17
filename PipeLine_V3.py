
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

from Pipe import optimize, generator,predictor
from Pipe import pso
from druida.tools.utils import CAD 


print('Check import functions: Done.')

# Arguments
parser = argparse.ArgumentParser()
# boxImagesPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Images Jorge Cardenas 512\\"
# DataPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Exports\\output\\"
# simulationData="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\DBfiles\\"

boxImagesPath="../../data/MetasufacesData/Images-512-Bands/"
#boxImagesPath="../../../data/MetasufacesData/Images-512-Suband/"
DataPath="../../data/MetasufacesData/Exports/output/"
simulationData="../../data/MetasufacesData/DBfiles/"
validationImages="../../data/MetasufacesData/testImages/"


Substrates={"Rogers RT/duroid 5880 (tm)":0, "other":1}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
TargetGeometries={"circ":[1,0,0],"box":[0,1,0], "cross":[0,0,1]}
Bands={"30-40":0,"40-50":1, "50-60":2,"60-70":3,"70-80":4, "80-90":5}



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
    parser.add_argument("-conditional_len_gen",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-conditional_len_pred",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-cond_channel",type=int)
    parser.add_argument("-metricType",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-latent",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-spectra_length",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-gen_model",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-one_hot_encoding",type=int)
    parser.add_argument("-output_path",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-working_path",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-gan_version",type=bool)
    parser.add_argument("-output_channels", type=int)
    parser.add_argument("-output_size",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-resnet_arch",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-validation_images",type=bool)
    parser.add_argument("-n_particles",type=int)



    parser.run_name = args["-run_name"]
    parser.epochs =  args["-epochs"]
    parser.batch_size = args["-batch_size"]
    parser.workers=args["-workers"]
    parser.gpu_number=args["-gpu_number"]
    parser.image_size = args["-image_size"]
    parser.dataset_path = args["-dataset_path"]
    parser.device = args["-device"]
    parser.learning_rate = args["-learning_rate"]
    parser.conditional_len_gen = args["-conditional_len_gen"] #Incliuding 3 top frequencies
    parser.conditional_len_pred = args["-conditional_len_pred"] #Incliuding 3 top frequencies
    parser.metricType= args["-metricType"] #this is to be modified when training for different metrics.
    parser.latent=args["-latent"] #this is to be modified when training for different metrics.
    parser.spectra_length=args["-spectra_length"] #this is to be modified when training for different metrics.
    parser.gen_model=args["-gen_model"]
    parser.one_hot_encoding = args["-one_hot_encoding"] #if used OHE Incliuding 3 top frequencies
    parser.output_path = args["-output_path"] #if used OHE Incliuding 3 top frequencies
    parser.working_path = args["-working_path"] #if used OHE Incliuding 3 top frequencies
    parser.gan_version=True
    parser.output_channels=args["-output_channels"]
    parser.output_size=args["-output_size"]
    parser.resnet_arch=args["-resnet_arch"]
    parser.cond_channel=args["-cond_channel"]
    parser.validation_images=args["-validation_images"]
    parser.n_particles=args["-n_particles"]

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
    
def cad_generation(images_folder,destination_folder,image_file_name,sustratoHeight):

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
    units="um"
    GoalSize=500
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


def prepare_data(files_name, device,df,classes,classes_types,z,substrate_encoder, materials_encoder,surfaceType_encoder,TargetGeometries_encoder,bands_encoder):
    bands_batch,array1,array_labels=[],[],[]
    a = []        

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
            #preparing data from spectra for each image
            data_raw=np.array(train.values.T)
            values=data_raw[1]
            all_frequencies=data_raw[0]
            
            all_frequencies_predictor = np.array([(float(i)-min(all_frequencies))/(max(all_frequencies)-min(all_frequencies)) for i in all_frequencies])
            all_frequencies = np.array([(float(i)-min(all_frequencies))/(max(all_frequencies)-min(all_frequencies)) for i in all_frequencies])

            data,fre_peaks, results_half =  find_peaks_func(values, all_frequencies)
            data_pred,fre_peaks_pred, results_half_pred =  find_peaks_func(values, all_frequencies_predictor)


            #labels_peaks=torch.cat((torch.from_numpy(data),torch.from_numpy(fre_peaks)),0)
            """this has 3 peaks and its frequencies-9 entries"""
            labels_peaks=torch.cat((torch.from_numpy(data),torch.from_numpy(fre_peaks),torch.from_numpy(results_half)),0)

            labels_tensor_truth=torch.cat((torch.from_numpy(data_pred),torch.from_numpy(fre_peaks_pred)),0)
            a.append(labels_tensor_truth)

            """This has geometric params 6 entries"""
            conditional_data,condition_predictor,sustratoHeight = set_conditioning(df,name,classes[idx],
                                                classes_types[idx],
                                                Bands[str(band_name)],
                                                None)



            bands_batch.append(band_name)

            #loading data to tensors for discriminator
            tensorA = torch.from_numpy(values) #Just have spectra profile
            labels = torch.cat((conditional_data.to(device),labels_peaks.to(device),tensorA.to(device))) #concat side    
            array_labels.append(labels) # to create stack of tensors
            #latent_tensor=torch.randn(1,parser.latent)
            latent_tensor=z

            if parser.gan_version:

                noise = torch.cat((noise.to(device),latent_tensor.to(device)))
            else:
        
                pass

    return array1, array_labels, noise,data_raw,sustratoHeight, a,condition_predictor

def find_peaks_func(values, all_frequencies):
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

    return data, fre_peaks, results_half

            
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
        sustratoHeight= sustratoHeight[-1]
        substrateWidth = 5 # 5 mm size
        

    # ---conditioning Generator
    values_array=torch.Tensor(geometry)
    values_array=torch.cat((values_array,torch.Tensor([sustratoHeight,substrateWidth,band ])),0)
    values_array = torch.Tensor(values_array)
    #contitioning Predictor
    values_array_predictor=torch.cat((torch.Tensor(geometry),torch.Tensor([band])),0)
    #values_array_pred = torch.stack(values_array_predictor)

    return values_array, values_array_predictor,sustratoHeight


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
        _, labels, noise,real_values,sustratoHeight = prepare_data(names, device,df,classes,classes_types,
                                                             None,
                                                             None,
                                                             None,
                                                             None,
                                                             None)

        testTensor = noise.type(torch.float).to(device)

        if parser.gan_version:
            label_conditions = torch.stack(labels).type(torch.float).to(device) #Discrminator Conditioning spectra
            noise = noise.type(torch.float).to(device) #Generator input espectro+ruido
            label_conditions = torch.nn.functional.normalize(label_conditions, p=2.0, dim=1, eps=1e-5, out=None)

            t_np = noise.cpu().numpy() #convert to Numpy array

            fake = netG.model(label_conditions,testTensor,parser.batch_size).detach().cpu()

        else:

            pass


        ##Eval

        """Saving Data"""

        if not os.path.exists(parser.output_path):
            os.makedirs(parser.output_path)

        with open(parser.output_path+"/usedmodel.txt", "a") as f:
            f.write("original batch:"+str(names))
            f.write("\n")

            f.write("image_size:"+str(parser.image_size))
            f.write("\n")
            
            f.write("condition_len:"+str(parser.conditional_len_gen))
            f.write("\n")

            f.write("latent:"+str(parser.latent))
            f.write("\n")

            f.write("spectra_length:"+str(parser.spectra_length))
            f.write("\n")

            f.write("gen_model:"+str(parser.gen_model))
            f.write("\n")
            
            f.write("one_hot_encoding:"+str(parser.one_hot_encoding))
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
        cad_generation(imagesFolder,destinationFolder,image_name,sustratoHeight)
        #plot Data

        break


def load_validation_data(device,z):

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
        _, labels, noise,real_values,sustratoHeight, truth,condition_predictor = prepare_data(names, device,df,classes,classes_types,z,
                                                             None,
                                                             None,
                                                             None,
                                                             None,
                                                             None)
        break

    
    return labels, noise,truth,condition_predictor,real_values,names

def opt_loop(device,generator,predictor,z,y_truth,conditions_predictor,conditions_generator):


    rng = np.random.default_rng()
    var_max  = [1 for _ in range(parser.latent)]
    var_min = [0 for _ in range(parser.latent)]
    z = z.detach().cpu().numpy()
    #take first original random Z
    #optimizing Z requires generating random particles considering the original
    #Z value.

    #var_max=var_max*z
    #ones= np.where((var_max)>1)
    #var_max[ones]=1

    #var_min=var_min*z
    #zeros= np.where((var_min)<0)
    #var_min[zeros]=0

    #Define swarm
    global swarm
    swarm = pso.Swarm(particles_number=parser.n_particles, variables_number=parser.latent, var_max=var_max, var_min=var_min)
    swarm.create()


    #first fitness value

    for index in range(len(swarm.particles)):
        particle = swarm.particles[index]
        fitness = particle_processing(device, 
                                      generator, 
                                      predictor,
                                      particle, 
                                      y_truth,
                                      conditions_predictor,
                                      conditions_generator)
        
        swarm.pbest[particle.id_] = fitness

    best_index=swarm.get_particle_best_fit(swarm.particles)
    #Optimization loop

    pi_best = swarm.particles.copy()#array initial particles

    for i in range(parser.epochs):
        print("current it.:"+str(i))
        particulas_anterior = []
        particulas_anterior  = swarm.particles.copy() #Array de particulas

        x , v = swarm.nuevas_particulas(particulas_anterior, pi_best, swarm.pg, swarm.velocidades,i)
        ### se actualizan las particulas originales con las nuevas particulas actualizadas
        for index_, particle in enumerate(x):
            swarm.particles[index_].values_array = particle.values_array

        swarm.velocidades = np.copy(v) #aquí un arreglo de vectores

        valores = np.zeros(parser.n_particles)
        
        ### se itera sobre cada particula y se simula

        for index in range(len(swarm.particles)):
            particle = swarm.particles[index]

            """possibly change conditioning vector
            [geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,band]
            """
            #conditioning[0] = random.choice([0,1,2]) #chosing between differente geometries
            fitness = particle_processing(device, 
                                          generator, 
                                          predictor,
                                          particle,
                                          y_truth,
                                          conditions_predictor,
                                          conditions_generator)
            valores[index]=fitness

            if valores[index] < swarm.pbest[index]:
                swarm.pbest[index] = valores[index]
                pi_best[index] = swarm.particles[index] #swarm particles is updated before with new particle

        if np.min(swarm.pbest) < swarm.gbest:
            swarm.gbest = np.min(swarm.pbest) #swarm.gbest comes from get_particle_best_fit
            swarm.pg = pi_best[np.argmin(swarm.pbest)].values_array
                
        best_index=np.argmin(swarm.pbest)
        
        print("best_particle", json.dumps(swarm.particles[best_index].values_array.tolist()))
    print("Minimo global encontrado: "+str(swarm.gbest))

    return swarm.particles[best_index].values_array

def particle_processing(device,generator,predictor, particle,  y_truth,conditions_predictor,conditions_generator):


    # ------ generate ------------
    fake = generator.model(conditions_generator,torch.from_numpy(particle.values_array).float().to(device),parser.batch_size)    
    y_predicted=predictor.model(input_=fake, conditioning=conditions_predictor.to(device) ,b_size=parser.batch_size)

    #take particle and generate with its vector
    fitness = pso.fitness(y_predicted,y_truth,0)
    fitness = fitness.detach().item()
    return fitness

def save_results(fitness, y_predicted,y_truth,fake,z,values_array,names):
    with open(parser.output_path+"/usedmodel.txt", "a") as f:
        f.write("batch:"+str(parser.run_name))
        f.write("\n")

        f.write("fitness:"+str(fitness))
        f.write("\n")
        
        f.write("Predicte:"+str(y_predicted))
        f.write("\n")

        f.write("latent:"+str(z))
        f.write("\n")

        f.write("Truth:"+str(y_truth))
        f.write("\n")

        f.write("conditioning:"+str(values_array))
        f.write("\n")

        f.write("conditioning:"+str(names))
        f.write("\n")


    f = open('data.json')
    data = json.load(f)

    spectra,frequencies=y_truth[1],y_truth[0]
    plt.plot(frequencies, spectra)
        # Saving figure by changing parameter values
    plt.savefig(parser.output_path+"/spectra_expected"+".png")

    
    # upscaling
    save_image(fake, parser.output_path+"/best_generated_image.png")
    image = cv2.imread(parser.output_path+"/best_generated_image.png")

    r = 512.0 / image.shape[1]
    dim = (512, int(image.shape[0] * r))
    # perform the actual resizing of the image using cv2 resize
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(parser.output_path+"/best_generated_image_512.png", resized) 

    imagesFolder = parser.output_path+"/"
    destinationFolder = parser.output_path+"/"
    image_name = "best_generated_image_512.png"
    time.sleep(3)


    """NOTE: subs height must be trained"""
    #1.575 0.787 0.508 0.252
    sustratoHeight=0.508
    cad_generation(imagesFolder,destinationFolder,image_name,sustratoHeight=sustratoHeight)
        


def main(args):

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    arguments(args)
    join_simulationData()  

    trainer = Stack.Trainer(parser)

    netG = generator.Generator(args=args)

    #--------load validation data
    z=torch.randn(parser.latent)
    labels, z, y_truth, condition_predictor,real_values,names=load_validation_data(device, z)
    label_conditions = torch.stack(labels).type(torch.float).to(device)
    noise = z.type(torch.float).to(device)
    label_conditions = torch.nn.functional.normalize(label_conditions, p=2.0, dim=1, eps=1e-5, out=None)

    # ------ generate ------------
    fake = netG.model(label_conditions,noise,parser.batch_size)
    
    if not os.path.exists(parser.output_path):
        os.makedirs(parser.output_path)

    save_image(fake.detach().cpu(), parser.output_path+"/generated_image.png")

    # ------- load predictor ----------
    predictor_obj = predictor.Predictor(args=args)

    if parser.validation_images:

        condition = torch.nn.functional.normalize(condition_predictor, p=2.0, dim=-1, eps=1e-5, out=None)
        y_predicted=predictor_obj.model(input_=fake, conditioning=condition.to(device) ,b_size=parser.batch_size)
        y_truth = torch.stack(y_truth).to(device)

        #------ Optimization process ------ 

        fitness = pso.fitness(y_predicted,y_truth,0)
        if fitness > 0.001:

            #optimize
            best_z = opt_loop(device=device, generator=netG,
                    predictor=predictor_obj,
                    z=noise,
                    y_truth=y_truth,
                    conditions_predictor=condition,
                    conditions_generator=label_conditions)
        else:
            best_z=z

    fake = netG.model(label_conditions,torch.from_numpy(best_z).float().to(device),parser.batch_size)

    y_predicted=predictor_obj.model(input_=fake, conditioning=condition.to(device) ,b_size=parser.batch_size)

    #take particle and generate with its vector
    fitness = pso.fitness(y_predicted,y_truth,0)
    fitness = fitness.detach().item()

    save_results(fitness, y_predicted,real_values,fake,best_z,condition,names)


if __name__ == "__main__":

    name = str(uuid.uuid4())[:8]

    #if not os.path.exists("output/"+str(name)):
    #        os.makedirs("output/"+str(name))
            
    args =  {"-gen_model":"models/NETGModelTM_abs__GANV2_128_FWHM_ADAM_16Sep.pth",
             "-pred_model":"models/trainedModelTM_abs__RESNET152_Bands_8sep_5e-4_500epc_ADAM_6out.pth",
             "-run_name":"GAN Training",
                                       "-epochs":50,
                                       "-batch_size":1,
                                       "-workers":1,
                                       "-gpu_number":1,
                                       "-image_size":128,
                                       "-dataset_path": os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/'),
                                       "-device":"cpu",
                                       "-learning_rate":5e-5,
                                       "-conditional_len_gen":15,
                                       "-conditional_len_pred":4,
                                       "-resnet_arch":"resnet152",
                                       "-metricType":"AbsorbanceTM",
                                       "-latent":112,
                                       "-n_particles":50,
                                       "-output_size":6,
                                       "-output_channels":3,
                                       "-spectra_length":100,
                                       "-one_hot_encoding":0,
                                       "-validation_images":True,
                                       "-gan_version":True,
                                       "-cond_channel":3,
                                       "-working_path":"./Generator_eval/",
                                       "-output_path":"./output/"+name} #OJO etepath lo lee el otro main

    main(args)

print('Check main function: Done.')