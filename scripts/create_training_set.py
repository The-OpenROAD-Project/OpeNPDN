#!/usr/bin/env python3

import sys
import subprocess
import numpy as np
from tqdm import tqdm 
sys.path.append('src')
from T6_PSI_settings import T6_PSI_settings
from scipy.stats import multivariate_normal
import math
settings_obj = T6_PSI_settings.load_obj();

start_map    = settings_obj.start_maps
num_maps  = settings_obj.num_maps
chip_size_x = math.floor(settings_obj.current_map_num_regions*settings_obj.WIDTH_REGION*1e6)
chip_size_y = math.floor(settings_obj.current_map_num_regions*settings_obj.LENGTH_REGION*1e6)

num_points = 1000;
size_gauss = 1.5;
scale = settings_obj.max_current;
chip_size = np.mean([chip_size_x, chip_size_y],0)
print("Generating random current maps for training ML model")
end_map = start_map + num_maps
for t in tqdm(range(start_map,end_map)):
    cur_map = np.zeros((chip_size_x, chip_size_y))
    for i in range(num_points):
        n = 2
        mu = np.reshape(chip_size_x*np.random.rand(1, n),n)
        #print(mu)
        #mu = np.random.random_sample((n,))
        #print(mu)
        sigma = np.triu(np.random.random_sample((n,n)))
        sigma = sigma + sigma.transpose()
        sigma = sigma + n*np.eye(n)
        sigma = chip_size*sigma*size_gauss;
        x1 = np.arange(chip_size_x)
        x2 = np.arange(chip_size_y)
        X1, X2 = np.meshgrid(x1,x2)
        X1=np.reshape(X1,(-1,1))
        X2=np.reshape(X2[:],(-1,1))
        X3 = np.hstack((X1,X2))
        F = multivariate_normal.pdf(X3, mu, sigma);
        #print(F.shape)
        if(np.random.rand(1)>0.5):
            cur_map = cur_map + np.reshape(F,(x2.shape[0], x1.shape[0]))
        else:
            cur_map = cur_map -np.reshape(F, (x2.shape[0], x1.shape[0]))
    min_cur_map = min(cur_map.flatten())
    max_cur_map = max(cur_map.flatten())
    cur_map = (1/(max_cur_map - min_cur_map))*(cur_map - min_cur_map)*scale
    cur_map = cur_map/ settings_obj.VDD
    file_name =  "./input/current_maps/current_map_%d.csv"%(t)
    with open(file_name, 'wb') as outfile:
        np.savetxt(outfile, cur_map, delimiter=',')

