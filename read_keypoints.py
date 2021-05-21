import pickle
import os
import shutil
import pandas as pd
from tqdm import tqdm

class ReadKeypoints:
    def __init__(self, el):
        self.el = el
        self.path = "./output/joint/detectron/"+el+"/"+el+"_DJ.pkl"
        with open(self.path, 'rb') as f:
            self.file = pickle.load(f)
        self.annot = pd.read_csv("./output/joint/detectron/"+el+"/"+el+"_labels.csv", sep='\t',header=None).values[1:-2]

    def is_equal_size(self):
        len_annot = len(self.annot)
        len_keyp = len(self.file)
        return len_annot == len_keyp

    # return dictionary containing the specific requested informations
    # key_name argument: array of string
    # pred_boxes // scores // pred_classes // pred_keypoints // pred_keypoint_heatmaps
    def get_features(self, keys_name):
        
        if not self.is_equal_size():
            print(self.el)
            print("ERROR: annotations and keypoints frame do not match - "+str(len(self.annot))+","+str(len(self.file))+"\nHINT: check on last two lines")
            exit()

        keys = {}
        i = 0
        for el in self.file:
            aux = {}
            for name in keys_name:
                aux[name] = el.get(name)
            aux['tag']=int(self.annot[i][2])
            keys[i] = aux
            i += 1
           
        return keys

    # print Instances info of the specific frame index: num_frame
    def print_file_content(self):
        print("printing...")
        print(type(self.file))
        print(len(self.file))
        # print(self.file[num_frame])

    def get_info(self):
        print(len(self.file))

if __name__ == '__main__':

    print("hello")
    video_list = os.listdir('./output/joint/detectron/')
    
    for el in tqdm(video_list):
        if el not in 'total_info':
            output_fname = "./output/joint/detectron/total_info/"+el+"_DJA.pkl"

            if not os.path.isfile(output_fname):
                read_keypoint = ReadKeypoints(el)
                total_info = read_keypoint.get_features(['scores', 'pred_keypoints'])
                with open(output_fname, 'wb') as handle:
                    pickle.dump(total_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
                del read_keypoint