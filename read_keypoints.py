import pickle
import os
import shutil

class ReadKeypoints:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'rb') as f:
            print("carico pickle")
            self.file = pickle.load(f)

    # return dictionary containing the specific requested informations
    # key_name argument: array of string
    # pred_boxes // scores // pred_classes // pred_keypoints // pred_keypoint_heatmaps
    def get_features(self, keys_name):
        keys = {}
        i = 0
        for el in self.file:
            aux = {}
            for name in keys_name:
                aux[name] = el.get(name)
            keys[i] = aux
            i += 1
            if i == 2:
                print(keys)
                exit()

        return keys

    # print Instances info of the specific frame index: num_frame
    def print_file_content(self, num_frame):
        print("printing...")
        print(type(self.file))
        print(len(self.file))
        print(self.file[num_frame])


if __name__ == '__main__':
    '''
    print("hello")
    path = "./output/joint/detectron/Abuse001_x264/Abuse001_x264_DJ.pkl"
    read_keypoint = ReadKeypoints(path)
    read_keypoint.get_features(['pred_keypoints'])
    '''
    for el in os.listdir('./output/joint/detectron/'):
        aux = os.listdir('./output/joint/detectron/' + el)
        if len(aux) < 2:
            print(el)
            shutil.rmtree('./output/joint/detectron/' + el)
