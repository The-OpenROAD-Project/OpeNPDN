#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import gstools as gs
#import csv
import numpy as np
from time import time
from tqdm import tqdm
import os

current_map_params = {
'unit_micron' : 2000
}

random_map_params = {
'x_dim' : 300,
'y_dim' : 300,
'var' : 1,
'max_cur' : 5e-7,
'len_scale' : 25,
'num_maps' : 1
}
random_map_params = {
'x_dim' : 300,
'y_dim' : 300,
'var' : 1,
'max_cur' : 1,
'len_scale' : 10,
'num_maps' : 1
}

class current_mapgen():  
  def __init__(self, current_map_params, random_map_params = None):
    self.current_map_params = current_map_params
    self.random_map_params = random_map_params
  
  def gen_random_maps(self, output_str=None, random_map_params = None):
    
    if random_map_params == None:
      if self.random_map_params == None:
        print("ERROR: random map parameters not specified")
        return 
      else:
        random_map_params = self.random_map_params
    if output_str is not None:
      directory = "./data"
      if not os.path.exists(directory):
          os.makedirs(directory)
    x_dim = int(random_map_params['x_dim'])
    y_dim = int(random_map_params['y_dim'])
    x = range(x_dim)
    y = range(y_dim)
    var = random_map_params['var']
    #max_cur = random_map_params['max_cur'] * (0.9 + 0.1*np.random.random())
    max_cur = random_map_params['max_cur']
    len_scale = random_map_params['len_scale']
    num_maps = random_map_params['num_maps']
    maps = np.zeros((0,x_dim,y_dim))
    for i in tqdm(range(num_maps)):
      model = gs.Gaussian(dim=2, var=var, len_scale=len_scale)
      srf = gs.SRF(model)
      fields=srf((x, y), mesh_type='structured')
      min_val = np.min(fields)
      max_val = np.max(fields)
      fields = max_cur * (fields - min_val)/(max_val - min_val)
      if output_str is not None:
        np.savetxt("./data/"+output_str+"_"+str(i)+".map", fields, delimiter=",")
      #fig,ax = plt.subplots()
      #im = ax.imshow(fields.T)
      #fig.colorbar(im)
      #plt.show(block = False)
      maps = np.append(maps,
                       fields[np.newaxis,...],
                       axis=0)
    return maps

  def to_dok_map(self,maps):
    unit_micron = self.current_map_params['unit_micron']
    #print(maps.shape)
    if maps.ndim == 2:
      maps_new = np.copy(maps)
      maps_new = maps_new[np.newaxis,...]
      #print(maps_new.shape)
    elif maps.ndim == 3:
      maps_new = maps
    else:
      print("ERROR: unexpected dimensions for maps")
      return 
    maps_dok = np.zeros((0,np.prod(maps_new[0,:].shape),3))
    for cur_map in tqdm(maps_new):
      #print(cur_map.shape)
      idx_arr = np.array(list(np.ndindex(cur_map.shape))) 
      #print(idx_arr)
      val_arr = cur_map[idx_arr[:,0],idx_arr[:,1]] 
      idx_arr = idx_arr * unit_micron
      map_dok = np.concatenate((idx_arr,
                                val_arr[:,np.newaxis]),axis =1)
      #print(maps_dok.shape)
      #print(map_dok.shape)
      maps_dok = np.append(maps_dok,map_dok[np.newaxis,:],axis=0)
      #print(maps_dok.shape)
    
    return maps_dok
      

if __name__ == '__main__':
  t0 = time()
  curr_mapgen_h = current_mapgen(current_map_params,random_map_params)
  #TODO current_map_processing VSRC processing
  st = time()
  print("start_time %f"%(st-t0))
  maps = curr_mapgen_h.gen_random_maps(output_str="test_current_map")
  t = time()
  print("create_maps time %f"%(t - st))
  maps_dok = curr_mapgen_h.to_dok_map(maps)
  #print(maps_dok)
  print("convert to dok %f"%(time() - t))
  plt.show()
  
