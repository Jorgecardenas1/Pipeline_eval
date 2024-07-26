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

# CAD

from druida.tools.utils import CAD 

parser = argparse.ArgumentParser()





class Generator:
    
    def __init__(self, args):
        self.args = args

        os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(device)

        self.arguments(self.args)

        trainer = Stack.Trainer(parser)
        # Sizes for discrimnator and generator
        """Z product"""
        input_size=parser.spectra_length+parser.condition_len

        """this for Z concat"""
        #input_size=parser.spectra_length+parser.condition_len+parser.latent
        
        generator_mapping_size=parser.image_size
        output_channels=3

        self.model = Stack.Generator(trainer.gpu_number, input_size, generator_mapping_size, output_channels)
        #self.model.load_state_dict(torch.load(parser.gen_model,map_location=torch.device(device)).state_dict())
        self.model.load_state_dict(torch.load(parser.gen_model,map_location=torch.device(device)))

        self.model.eval()
        
        if device.type!='cpu':
            self.model.cuda()

    def generate(conditions, data, noise):
        pass



    def arguments(self,args):
    
        parser.add_argument("-run_name",type=str)
        parser.add_argument("-epochs",type=int)
        parser.add_argument("-batch_size",type=int)
        parser.add_argument("-workers",type=int)
        parser.add_argument("-gpu_number",type=int)
        parser.add_argument("-image_size",type=int)
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


        parser.run_name = args["-run_name"]
        parser.epochs =  args["-epochs"]
        parser.batch_size = args["-batch_size"]
        parser.workers=args["-workers"]
        parser.gpu_number=args["-gpu_number"]
        parser.image_size = args["-image_size"]
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

