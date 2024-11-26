
import os
import glob
from druidaHFSS.modules import tools
from scipy.signal import find_peaks,peak_widths
import numpy as np
import pandas as pd

image_size=512
imagesPath="./data/MetasurfacesData/Images"
DataPath="./data/MetasurfacesData/Exports/output/"

folders=glob.glob(imagesPath+"/*/", recursive = True)
files=[]

processed="/processed512/"
Bands={"75-78":0}
absortionLevel = 0.6
bands = [
    [75,78]
]


print(folders)

for folder in folders:
    
    if folder != imagesPath+processed:
        files=(files+glob.glob(folder+"/*"))


for file in files:
    for band in bands:

        fileName_absolute = os.path.basename(file) 
        newname=fileName_absolute.split(".")[0]+"_"+str(band[0])+"-"+str(band[1])+".png"
        path=os.path.dirname(file)

        #ROI is 
        
        
        for file_name in glob.glob(DataPath+fileName_absolute.split("_")[5]+'/files/'+'/'+'AbsorbanceTM'+'*'+fileName_absolute.split("_")[6].split(".")[0]+'.csv'): 
            #loading the absorption data
            train = pd.read_csv(file_name)

            # # the band is divided in chunks 
            if Bands["75-78"]==0:
                
                train=train.loc[1:100]

            elif Bands["75-78"]==1:
                
                train=train.loc[101:200]
            
            
            #preparing data from spectra for each image
            data=np.array(train.values.T)
            values=data[1]
            all_frequencies=data[0]
            all_frequencies = np.array([(float(i)-min(all_frequencies))/(max(all_frequencies)-min(all_frequencies)) for i in all_frequencies])

            #get top freqencies for top values 
            peaks = find_peaks(values, threshold=0.00001)[0] #indexes of peaks
            results_half = peak_widths(values, peaks, rel_height=0.5) #4 arrays: widths, y position, initial and final x
            results_half = results_half[0]
            data = values[peaks]
            fre_peaks = all_frequencies[peaks]
            
            length_output=3

            if len(peaks)>length_output:
                if max(data)>absortionLevel:
                    image_rgb=tools.cropImage( file,image_path=processed,
                               image_name=newname,
                               output_path=imagesPath, 
                               resize_dim=(image_size,image_size))

            elif len(peaks)==0:

                pass

            else:
                if max(data)>absortionLevel:
                    image_rgb=tools.cropImage( file,image_path=processed,
                               image_name=newname,
                               output_path=imagesPath, 
                               resize_dim=(image_size,image_size))
