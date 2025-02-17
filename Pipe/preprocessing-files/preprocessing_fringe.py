
import os
import glob
from pathlib import Path
import random

from PIL import Image

from druida import Stack
from druida import setup
import numpy as np
import pandas as pd
from druida.tools import utils
from tqdm.notebook import tqdm
import torch
import json
import torchvision.transforms as transforms

from torchvision.utils import save_image


image_size=512

imagesPath="./data/MetasurfacesDataV3/Images/processed512"
imagesDestination="./data/MetasurfacesDataV3/Images-512-Bands"
testDestination="./data/MetasurfacesDataV3/testImages"
boxImagesPath="./data/MetasurfacesDataV3/Images-512-Bands/"
simulationData="./data/MetasurfacesDataV3/DBfiles/"

Bands={"75-78":0}


def join_simulationData():
    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    
    df.to_csv('out.csv',index=False)

join_simulationData()

df = pd.read_csv("out.csv")

folders=glob.glob(imagesPath+"/*.png", recursive = True)
random.shuffle(folders)


def fringe(image, value):
    fringe_width = 1

    # Get image dimensions
    C, H, W = image.shape
    # Create a mask for the fringe
    mask = torch.zeros((H, W), dtype=torch.bool)

    # Top fringe
    mask[:fringe_width, :] = True
    # Bottom fringe
    mask[-fringe_width:, :] = True
    # Left fringe
    mask[:, :fringe_width] = True
    # Right fringe
    mask[:, -fringe_width:] = True



    # Expand the mask to match the image dimensions (C, H, W)
    mask = mask.unsqueeze(0).expand(1, -1, -1)
    
    # Change pixel values in the fringe (e.g., set to red [1, 0, 0])
    #fringe_color = torch.tensor([1, 0, 0]).view(3, 1, 1)  # RGB (normalized to [0, 1])
    """estos rangos son para cubrir el ancho de la celda para todos los tipos
    de celda que hay"""
    factor = ((value - 4.85) / (5.3 - 4.85)) 
    #print(value)

    image[2, mask[0]] = torch.tensor([[factor]])
            
    return image


transform_to_tensor = transforms.ToTensor()  # Converts to a tensor with values in [0, 1]


for folder in folders:

    image = Image.open(folder)

    fileName_absolute = os.path.basename(folder) 
    print(fileName_absolute)
    type_=fileName_absolute.split("_")[0]
    serial=fileName_absolute.split("_")[5]
    iter=fileName_absolute.split("_")[6].split("-")[1]
    row = df[(df['sim_id'] == serial) & (df['iteration'] == int(iter))]

    subtratewidth = json.loads(row["paramValues"].values[0])

    print(type_)
    if type_ == "cross":
        subtratewidth = float(subtratewidth[4])

    elif type_=="circ":
        subtratewidth = float(subtratewidth[3])
    elif type_=="splitcross":
        subtratewidth = float(subtratewidth[5])
    elif type_=="ring":
        subtratewidth = float(subtratewidth[5])

    else:
        pass


    tensor_image = transform_to_tensor(image) # Shape: (C, H, W)

    image = fringe(tensor_image, subtratewidth)
   
    save_image(tensor_image, folder)
