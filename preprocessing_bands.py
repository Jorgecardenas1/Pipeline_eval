
import os
import glob
from pathlib import Path
import random

from druida import Stack
from druida import setup
import numpy as np
import pandas as pd
from druida.tools import utils
from tqdm.notebook import tqdm


image_size=512

imagesPath="./data/MetasurfacesData/Images/processed512"
imagesDestination="./data/MetasurfacesData/Images-512-Bands"
testDestination="./data/MetasurfacesData/testImages"
boxImagesPath="./data/MetasurfacesData/Images-512-Bands/"
Bands={"75-78":0}

folders=glob.glob(imagesPath+"/*.png", recursive = True)
random.shuffle(folders)
size = len(folders)
trainLen = int(size*0.95)
testLen= int(size*0.05)

training = folders[0:trainLen]
testing = folders[trainLen:]
files=[]

"""
1. crear los folder en la subcarpeta previamente
2. el folder circle, se nombre "circ" y luego se le cambia el nombre.
"""


index=0
for folder in training:
    fileName_absolute = os.path.basename(folder) 
    type_=fileName_absolute.split("_")[0]

    os.chmod(os.path.abspath(folder), 0o755)
    os.rename(os.path.abspath(folder), os.path.abspath(imagesDestination+"/"+type_+"/"+fileName_absolute) ) 
    index+=1
    print(index)

index=0
for folder in testing:
    fileName_absolute = os.path.basename(folder) 
    type_=fileName_absolute.split("_")[0]

    os.chmod(os.path.abspath(folder), 0o755)
    os.rename(os.path.abspath(folder), os.path.abspath(testDestination+"/"+type_+"/"+fileName_absolute) ) 
    index+=1
    print(index)
 
