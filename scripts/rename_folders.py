import os
import pickle
import sys
from os.path import isdir, isfile, join

root_dir = sys.argv[1]

with open('imagenet_folder_to_cls.pkl', 'rb') as f:
    class2idx = pickle.load(f)

for key, value in class2idx.items():
    #print(root_dir+'/'+str(value)+'/', root_dir+'/'+key+'/')
    os.rename(root_dir + '/' + str(value) + '/', root_dir + '/' + key + '/')
