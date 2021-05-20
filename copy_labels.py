import argparse
import cv2
import os
import pickle
from shutil import copyfile

from detectron2.utils.logger import setup_logger
from detectron2.utils.video_visualizer import VideoVisualizer

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from tqdm import tqdm
import subprocess

if __name__ == '__main__':
    path_in = "../video_tagger/output/"
    path_out = "./output/joint/detectron/"
    list_video = os.listdir(path_in+'video')

    for el in tqdm(list_video):
        file_name = el[:-4]
        
        if os.path.isfile(path_in+'annotation/'+file_name+'_labels.csv') and os.path.isdir(path_out+file_name+'/'):
            #print(path_in+'annotation/'+file_name+'_labels.csv \t to \t '+path_out+file_name+'/'+file_name+'_labels.csv')
            copyfile(path_in+'annotation/'+file_name+'_labels.csv', path_out+file_name+'/'+file_name+'_labels.csv')
        
