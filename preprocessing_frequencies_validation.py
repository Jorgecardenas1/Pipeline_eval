
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


imagesPath="./data/MetasurfacesData/Images/processed512"
imagesDestination="./data/MetasurfacesData/Images-512-Bands"
testDestination="./data/MetasurfacesData/testImages"
boxImagesPath="./data/MetasurfacesData/Images-512-Bands/"
Bands={"75-78":0}

 

def join_imagesData():
    df = pd.DataFrame(columns=('path', 'class','class_target', 'freq','band'))

    namesArr = []
    classesArr = []
    targets = []
    dataloader = utils.get_data_with_labels(128,128,1, 
                                                testDestination,64,
                                                drop_last=False,
                                                filter="30-40")#filter disabled

    for data in tqdm(dataloader):
        _, classes, names, classes_types = data
        namesArr = namesArr + list(names)
        classesArr = classesArr + list(classes_types)
        targets = targets + list(classes_types)
    for item in Bands:
        for freq in range(0,100):
            for idx,name in enumerate(namesArr):
                df = pd.concat([
                df, pd.DataFrame([[name,classesArr[idx], targets[idx],freq,item]], columns=['path', 'class','class_target', 'freq','band'])]
                )

    df.to_csv('outImagesValidation.csv',index=False)


join_imagesData()
