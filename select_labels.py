import argparse
import cv2
import os
import pickle
from shutil import copyfile, move
from tqdm import tqdm
import subprocess
import pandas as pd

if __name__ == '__main__':
    
    violent_video = []
    video_list = os.listdir('./output/joint/detectron/')
    for el in tqdm(video_list):
        if os.path.isfile('./output/joint/detectron/'+el+'/'+el+'_labels.csv') and el not in 'Panic.Cam1':
            aux = pd.read_csv('./output/joint/detectron/'+el+'/'+el+'_labels.csv', header=0, names=['name','tag'], sep='\t')
            if len(aux['tag']==1)>0:
                violent_video.append(el)
    print(len(violent_video))