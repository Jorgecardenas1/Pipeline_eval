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

Substrates={"Rogers RT/duroid 5880 (tm)":0, "other":1}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
TargetGeometries={"circ":0,"box":1, "cross":2}
Bands={"30-40":0,"40-50":1, "50-60":2,"60-70":3,"70-80":4, "80-90":5}

optimizing=False

parser = argparse.ArgumentParser()


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
        
    #conditions
    values_array=torch.Tensor([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,band])
    
    #building requested spectra
    x = np.array(data["band"].split("-"))
    upper = float(x[1])
    lower = float(x[0])
    array_frequs =  torch.tensor(np.arange(lower+0.1, upper+0.1, 0.1),dtype=torch.float32)

    spectra = torch.zeros(100)

    filtered = torch.isin(array_frequs,torch.tensor(data["frequencies"]))
    indexes = torch.where(filtered == True)[0]

    #this is the requeste spectra by the user
    spectra[indexes] = torch.tensor(data["values"])

    return values_array,spectra

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

def opt_loop():
    var_max  = [random.uniform(0.8,0.9) for _ in range(parser.latent)]
    var_min = [random.uniform(0.0,0.2) for _ in range(parser.latent)]

    swarm = pso.Swarm(particles_number=5, variables_number=parser.latent, var_max=var_max, var_min=var_min)
    swarm.create()
    


def main(args):

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #load args
    arguments(args)

    #load generator
    #generator_obj = generator.Generator(args=args)

    #load user request
    values_array,spectra = load_request()

    if optimizing:
        pass
    else:
        z=torch.rand(parser.latent)

    input_tensor = prepare_data(device,spectra,values_array,z)
    
    #generate
            
    #fake = generator_obj.model(input_tensor).detach().cpu()
    
    if not os.path.exists(parser.output_path):
        os.makedirs(parser.output_path)

    #save_image(fake, parser.output_path+"/generated_image.png")

    #load predictor
    predictor_obj = predictor.Predictor(args=args)
    
    #y_predicted=predictor_obj.model(input_=fake, conditioning=values_array.to(device) ,b_size=parser.batch_size)
    #print(y_predicted)

    #loss 


    #optimize
    opt_loop()
    #save results


    pass

if __name__ == "__main__":

    name = str(uuid.uuid4())[:8]

    #if not os.path.exists("output/"+str(name)):
    #        os.makedirs("output/"+str(name))
            
    args =  {"-gen_model":"models/NETGModelTM_abs__GAN_Bands_15May_110epc_64_6conds_zprod.pth",
             "-pred_model":"models/trainedModelTM_abs__RESNET152_Bands_11May_2e-5_100epc_h1000_f1000_64_MSE_arrayCond_2TOP2Freq.pth",
             "-run_name":"FullPipe",
             "-epochs":1,
             "-batch_size":1,
             "-workers":1,
             "-gpu_number":1,
             "-image_size":64,
             "-learning_rate":1e-5,
             "-device":"cpu",
             "-metricType":"AbsorbanceTM",
             "-cond_channel":3,
             "-condition_len":6,
             "-resnet_arch":"resnet152",
             "-latent":106,
             "-spectra_length":100,
             "-one_hot_encoding":0,
             "-working_path":"./Generator_eval/",
             "-output_path":"./output/"+name} #OJO etepath lo lee el otro main

    main(args)

print('Check main function: Done.')
