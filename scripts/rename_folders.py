

import os, sys, pickle
from os.path import isfile, isdir, join

root_dir = sys.argv[1]

with open('imagenet_folder_to_cls.pkl','rb') as f:
    class2idx = pickle.load(f)


for key, value in class2idx.items(): 
    #print(root_dir+'/'+str(value)+'/', root_dir+'/'+key+'/')
    os.rename(root_dir+'/'+str(value)+'/', root_dir+'/'+key+'/')
