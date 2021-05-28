from typing import Mapping


import pickle

if __name__ == '__main__':
    path = './output/joint/detectron/total_info/V_70_DJA.pkl'
    with open(path, 'rb') as f:
        file = pickle.load(f)
    print(file[0])