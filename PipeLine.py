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


from Pipe import optimize, generator


def main(args):

    #load generator
    generator_obj = generator.Generator(args=args)
    print(generator_obj.model)
    #load predictor

    #load user request
    #generate
    #predict
    #optimize

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
             "-output_path":"../output/"+name} #OJO etepath lo lee el otro main

    main(args)

print('Check main function: Done.')