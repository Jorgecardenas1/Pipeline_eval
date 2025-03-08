"""predictor.py: Predictor as class to load trained model."""
__author__      = "JORGE H. CARDENAS"
__copyright__   = "2024,2025"
__version__   = "1.0"

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

import glob
from tqdm.notebook import tqdm
import argparse
import json
from PIL import Image
import cv2 

parser = argparse.ArgumentParser()



class Predictor:

    def __init__(self, args):
        self.args = args
        self.arguments(self.args)

        os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model=self.get_net_resnet(self.device,hiden_num=600,dropout=0.3,features=400, Y_prediction_size=parser.output_size)
        #self.model = self.model.to(self.device)

        self.fringe_model = self.get_fringe_resnet(self.device)
        #self.fringe_model = self.fringe_model.to(self.device)

    def get_net_resnet(self,device,hiden_num=800,dropout=0.3,features=400, Y_prediction_size=100):
        
        model = Stack.Predictor_RESNET_V2(parser.resnet_arch,conditional=True, 
                                       cond_input_size=parser.conditional_len_pred, 
                                       cond_channels=parser.cond_channel, 
                                ngpu=1, image_size=parser.predictor_image_size ,
                                output_size=8, channels=3,
                                features_num=features,hiden_num=hiden_num, #Its working with hiden nums. Features in case and extra linear layer
                                dropout=dropout, 
                                Y_prediction_size=Y_prediction_size) #size of the output vector in this case frenquency points


        model.load_state_dict(torch.load(parser.pred_model,map_location=torch.device(device)))
        #model.load_state_dict(torch.load(parser.pred_model,map_location=torch.device(device)).state_dict())

        model.eval()

        if device.type!='cpu':
            model.cuda()


        return model
    
    def get_fringe_resnet(self,device):
        
        fringe_model = Stack.Fringe_RESNET_V2("resnet18",conditional=False, ngpu=1, image_size=45 ,
                                output_size=8, channels=1,
                                features_num=100,hiden_num=200, #Its working with hiden nums. Features in case and extra linear layer
                                dropout=0.4, 
                                Y_prediction_size=1) #size of the output vector in this case frenquency points
    
        fringe_model.load_state_dict(torch.load(parser.fringe_model,map_location=torch.device(device)))
        #finge_model.load_state_dict(torch.load(parser.pred_model,map_location=torch.device(device)).state_dict())

        fringe_model.eval()

        if device.type!='cpu':
            fringe_model.cuda()


        return fringe_model

    def arguments(self,args):
    
        parser.add_argument("-run_name",type=str)
        parser.add_argument("-epochs",type=int)
        parser.add_argument("-batch_size",type=int)
        parser.add_argument("-workers",type=int)
        parser.add_argument("-gpu_number",type=int)
        parser.add_argument("-image_size",type=int)
        parser.add_argument("-predictor_image_size",type=int)
        parser.add_argument("-device",type=str)
        parser.add_argument("-learning_rate",type=float)
        parser.add_argument("-conditional_len_pred",type=float) #This defines the length of our conditioning vector
        parser.add_argument("-metricType",type=str) #This defines the length of our conditioning vector
        parser.add_argument("-latent",type=int) #This defines the length of our conditioning vector
        parser.add_argument("-spectra_length",type=int) #This defines the length of our conditioning vector
        parser.add_argument("-pred_model",type=str) #This defines the length of our conditioning vector
        parser.add_argument("-one_hot_encoding",type=int)
        parser.add_argument("-output_path",type=str) #This defines the length of our conditioning vector
        parser.add_argument("-working_path",type=str) #This defines the length of our conditioning vector
        parser.add_argument("-cond_channel",type=int) #This defines the length of our conditioning vector
        parser.add_argument("-resnet_arch",type=str) #This defines the length of our conditioning vector
        parser.add_argument("-output_size",type=float) #This defines the length of our conditioning vector
        parser.add_argument("-fringe_model",type=str) #This defines the length of our conditioning vector



        parser.run_name = args["-run_name"]
        parser.epochs =  args["-epochs"]
        parser.batch_size = args["-batch_size"]
        parser.workers=args["-workers"]
        parser.gpu_number=args["-gpu_number"]
        parser.image_size = args["-image_size"]
        parser.predictor_image_size = args["-predictor_image_size"]
        parser.device = args["-device"]
        parser.learning_rate = args["-learning_rate"]
        parser.conditional_len_pred = args["-conditional_len_pred"] #Incliuding 3 top frequencies
        parser.metricType= args["-metricType"] #this is to be modified when training for different metrics.
        parser.latent=args["-latent"] #this is to be modified when training for different metrics.
        parser.spectra_length=args["-spectra_length"] #this is to be modified when training for different metrics.
        parser.pred_model=args["-pred_model"]
        parser.one_hot_encoding = args["-one_hot_encoding"] #if used OHE Incliuding 3 top frequencies
        parser.output_path = args["-output_path"] #if used OHE Incliuding 3 top frequencies
        parser.working_path = args["-working_path"] #if used OHE Incliuding 3 top frequencies
        parser.resnet_arch= args["-resnet_arch"] #this is to be modified when training for different metrics.
        parser.output_size = args["-output_size"] #Incliuding 3 top frequenciess
        parser.cond_channel = args["-cond_channel"] #if used OHE Incliuding 3 top frequencies
        parser.fringe_model=args["-fringe_model"]
