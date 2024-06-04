import glob as glob
import sys
import os
import time
#from Utilities.SaveAnimation import Video
from typing import Sequence, Optional, TypeVar
import uuid

from druida import Stack
from druida import setup
from druida.DataManager import datamanager
from druida.tools import utils

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

import glob
from tqdm.notebook import tqdm
import argparse
import json
from PIL import Image
import cv2 
import json


from Pipe import optimize, generator,predictor
from Pipe import pso
from druida.tools.utils import CAD 



Substrates={"Rogers RT/duroid 5880 (tm)":0, "other":1}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
TargetGeometries={"circ":0,"box":1, "cross":2}
Bands={"30-40":0,"40-50":1, "50-60":2,"60-70":3,"70-80":4, "80-90":5}

optimizing=False

parser = argparse.ArgumentParser()


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



def load_request():

    f = open('data.json')
    data = json.load(f)

    target_val=data["target"]
    category=data["category"]
    geometry=TargetGeometries[category]
    band=Bands[data["band"]]
    """"
    surface type: reflective, transmissive
    layers: conductor and conductor material / Substrate information
    """
    surfacetype=Surfacetypes[data["surfacetype"]]
    materialconductor=Materials[data['conductor']]
    materialsustrato=Substrates[data['substrate']]
    sustratoHeight= data["substrateHeight"]
    SubsLength=data["substrateLength"]

    #conditions
    values_array=torch.Tensor([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,SubsLength,band])
    
    spectra,_ = spectra_builder(data)
    y_truth = torch.concat( (torch.tensor(data["values"]), torch.tensor(data["frequencies"])),0)


    return values_array,spectra, y_truth

def spectra_builder(data):
    #building requested spectra
    x = np.array(data["band"].split("-"))
    upper = float(x[1])
    lower = float(x[0])
    array_frequs =  torch.tensor(np.arange(lower+0.1, upper+0.1, 0.1),dtype=torch.float32)

    spectra = torch.zeros(100)

    filtered = torch.isin(array_frequs,torch.tensor(data["frequencies"]))
    indexes = torch.where(filtered == True)[0]
    #this is the requested spectra by the user
    spectra[indexes] = torch.tensor(data["values"])
    return spectra,array_frequs

def arguments(args):

    
    parser.add_argument("-run_name",type=str)
    parser.add_argument("-epochs",type=int)
    parser.add_argument("-batch_size",type=int)
    parser.add_argument("-workers",type=int)
    parser.add_argument("-gpu_number",type=int)
    parser.add_argument("-image_size",type=int)
    parser.add_argument("-device",type=str)
    parser.add_argument("-learning_rate",type=float)
    parser.add_argument("-condition_len",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-n_particles",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-metricType",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-latent",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-cond_channel",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-spectra_length",type=int) #This defines the length of our conditioning vector
    parser.add_argument("-gen_model",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-pred_model",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-one_hot_encoding",type=int)
    parser.add_argument("-output_path",type=str) #This defines the length of our conditioning vector
    parser.add_argument("-working_path",type=str) #This defines the length of our conditioning vector


    parser.run_name = args["-run_name"]
    parser.epochs =  args["-epochs"]
    parser.batch_size = args["-batch_size"]
    parser.workers=args["-workers"]
    parser.gpu_number=args["-gpu_number"]
    parser.image_size = args["-image_size"]
    parser.device = args["-device"]
    parser.learning_rate = args["-learning_rate"]
    parser.condition_len = args["-condition_len"] #Incliuding 3 top frequencies
    parser.cond_channel = args["-cond_channel"] #Incliuding 3 top frequencies
    parser.n_particles = args["-n_particles"] #Incliuding 3 top frequencies

    parser.metricType= args["-metricType"] #this is to be modified when training for different metrics.
    parser.latent=args["-latent"] #this is to be modified when training for different metrics.
    parser.spectra_length=args["-spectra_length"] #this is to be modified when training for different metrics.
    parser.gen_model=args["-gen_model"]
    parser.pred_model=args["-pred_model"]
    parser.one_hot_encoding = args["-one_hot_encoding"] #if used OHE Incliuding 3 top frequencies
    parser.output_path = args["-output_path"] #if used OHE Incliuding 3 top frequencies
    parser.working_path = args["-working_path"] #if used OHE Incliuding 3 top frequencies

    print('Check arguments function: Done.')


def prepare_data(device,spectra,conditional_data,latent_tensor):
    noise = torch.Tensor()

    labels = torch.cat((conditional_data.to(device),spectra.to(device))) #concat side

    """multiply noise and labels to get a single vector"""
    tensor1=torch.mul(labels.to(device),latent_tensor.to(device) )

    """concat noise and labels adjacent"""
    #tensor1 = torch.cat((conditional_data.to(device),tensorA.to(device),latent_tensor.to(device),)) #concat side
    #un vector que inclue
    #datos desde el dataset y otros datos aleatorios latentes.

    """No lo veo tan claro pero es necesario para pasar los datos al ConvTranspose2d"""
    tensor2 = tensor1.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
    tensor3 = tensor2.permute(1,0,2,3)
    noise = torch.cat((noise.to(device),tensor3.to(device)),0)

    testTensor = noise.type(torch.float).to(device)

        ##Eval

    return testTensor



def opt_loop(device,generator,predictor,conditioning,spectra,z,y_predicted,y_truth):
    
    var_max  = [random.uniform(1.,1.3) for _ in range(parser.latent)]
    var_min = [random.uniform(0.5,0.99) for _ in range(parser.latent)]
    z = z.detach().cpu().numpy()

    #take first original random Z
    #optimizing Z requires generating random particles considering the original
    #Z value.

    var_max=var_max*z
    ones= np.where((var_max)>1)
    var_max[ones]=1

    var_min=var_min*z
    zeros= np.where((var_min)<0)
    var_min[zeros]=0

    #Define swarm
    global swarm
    swarm = pso.Swarm(particles_number=parser.n_particles, variables_number=parser.latent, var_max=var_max, var_min=var_min)
    swarm.create()


    #first fitness value

    for index in range(len(swarm.particles)):
        particle = swarm.particles[index]
        fitness = particle_processing(device, generator, predictor,particle, conditioning, spectra, y_truth)
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
            fitness = particle_processing(device, generator, predictor,particle, conditioning, spectra, y_truth)
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

def particle_processing(device,generator,predictor, particle, conditioning, spectra, y_truth):

    input_tensor = prepare_data(device,spectra,conditioning,torch.tensor(particle.values_array))
    fake = generator.model(input_tensor).detach().cpu()
    y_predicted=predictor.model(input_=fake, conditioning=conditioning.to(device) ,b_size=parser.batch_size)

    #take particle and generate with its vector
    fitness = pso.fitness(y_predicted,y_truth,0)

    return fitness

def save_results(fitness, y_predicted,y_truth,fake,z,values_array):
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


    f = open('data.json')
    data = json.load(f)

    spectra,frequencies=spectra_builder(data)
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

    #load args
    arguments(args)

    #load generator
    generator_obj = generator.Generator(args=args)

    #load user request
    values_array,spectra,y_truth = load_request()

    if optimizing:
        pass
    else:
        z=torch.rand(parser.latent)

    input_tensor = prepare_data(device,spectra,values_array,z)
    
    #generate
    fake = generator_obj.model(input_tensor).detach().cpu()
    
    if not os.path.exists(parser.output_path):
        os.makedirs(parser.output_path)

    save_image(fake, parser.output_path+"/generated_image.png")

    #load predictor
    predictor_obj = predictor.Predictor(args=args)
    y_predicted=predictor_obj.model(input_=fake, conditioning=values_array.to(device) ,b_size=parser.batch_size)

    #loss 
    fitness = pso.fitness(y_predicted,y_truth,0)
    if fitness > 0.01:

        #optimize
        best_z = opt_loop(device=device, generator=generator_obj,
                predictor=predictor_obj,
                conditioning=values_array,
                spectra=spectra,
                z=z,
                y_predicted=y_predicted,
                y_truth=y_truth)
    

    #final generation
    input_tensor = prepare_data(device,spectra,values_array,torch.tensor(best_z))
    
    #generate
    fake = generator_obj.model(input_tensor).detach().cpu()
    
    if not os.path.exists(parser.output_path):
        os.makedirs(parser.output_path)


    #predictor
    y_predicted=predictor_obj.model(input_=fake, conditioning=values_array.to(device) ,b_size=parser.batch_size)

    #loss 
    fitness = pso.fitness(y_predicted,y_truth,0)
    print("final fitness:",fitness)
    #save results
    save_results(fitness, y_predicted,y_truth,fake,z,values_array)

    

if __name__ == "__main__":

    name = str(uuid.uuid4())[:8]

    #if not os.path.exists("output/"+str(name)):
    #        os.makedirs("output/"+str(name))
            
    args =  {"-gen_model":"models/NETGModelTM_abs__GAN_Bands_1June_110epc_64_7conds_zprod.pth",
             "-pred_model":"models/trainedModelTM_abs__RESNET152_Bands_1June_2e-5_100epc_h1000_f1000_64_MSE_3top3freq.pth",
             "-run_name":name,
             "-epochs":30,
             "-batch_size":1,
             "-workers":1,
             "-gpu_number":1,
             "-image_size":64,
             "-learning_rate":1e-5,
             "-device":"cpu",
             "-metricType":"AbsorbanceTM",
             "-cond_channel":3,
             "-condition_len":7,
             "-n_particles":10,
             "-resnet_arch":"resnet152",
             "-latent":107,
             "-spectra_length":100,
             "-one_hot_encoding":0,
             "-working_path":"./Generator_eval/",
             "-output_path":"./output/"+name} #OJO etepath lo lee el otro main

    main(args)

print('Check main function: Done.')
