import json
import logging
import numpy as np
from current_mapgen import current_mapgen
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import scipy
import os

def inference(grid_params_file,
      tl_chk_pt,
      syn_chk_pt,
      log_level,
      data_dir,
      logger_file=None):
  global logger_h
  logger_h = create_logger(log_level, logger_file)
  plot_maps = logger_h.getEffectiveLevel()<=logging.DEBUG

  logger_h.info("Starting TL data generation.")
  set_globals(grid_params_file, ir_params_file)
  logger_h.info("Loaded global parameters:")
  logger_h.info("  Grid parameters: %s"%grid_params_file)
  logger_h.info("  IR solver parameters: %s"%ir_params_file)
  logger_h.info("  Templates: %s"%template_file)
  logger_h.debug("Loaded json value:%d"%grid_params['size_x'])
  logger_h.info("Generating the input data.")
  #'bp' 32
  designs =\
  ['aes','ibex','jpeg','dynamic_node','bp_fe','bp_be','bp_multi','swerv']
  cur_scale =\
  [   15,    10,    10,            15,     30,     35,        15,   12.5]
  inf_data = load_data('./designs', designs, cur_scale,plot_maps)
  #current_maps, current_maps_dok, signal_congestion = generate_maps(plot= plot_maps)
  logger_h.info("Running Inference.")
  TL_dataset = OpeNPDNDataset(list(inf_data.values()), 
                              labels_present=False,
                              normalize=False)
  TL_dataset.load_normalize(os.path.dirname(syn_chk_pt), True)
  set_global_hyperparams()
  TL_model = train_loop(TL_dataset, syn_chk_pt, plot_data)



  if plot_maps:
    plt.show()

def load_data(data_dir, designs, cur_scale, plot):
  inf_data ={}
  for n,cur_design in enumerate(tqdm(designs)):
    current_maps = np.zeros((0,3*region_size,3*region_size))
    cong_maps    = np.zeros((0,3*region_size,3*region_size)) 
    macro_maps   = np.zeros((0,3*region_size,3*region_size))
    eff_dist_maps= np.zeros((0,3*region_size,3*region_size))
    current_map = np.loadtxt('%s/%s/current.csv'%(data_dir,cur_design),delimiter=',')
    current_map = scipy.ndimage.gaussian_filter(current_map, 1.5)
    current_map = current_map*cur_scale[n]
    x_dim = current_map.shape[-2]
    y_dim = current_map.shape[-1]
    num_x,mod_x = divmod(x_dim,region_size)
    num_y,mod_y = divmod(y_dim,region_size)
    if mod_x>region_size/2:
        num_x+=1
    if mod_y>region_size/2:
        num_y+=1
    logger_h.debug("Design: %s x:%d y:%d"%(cur_design,num_x,num_y))
    start,end = end, end + num_x*num_y
    if plot and n==0:
      plt.figure()
      plt.imshow(current_map.T,cmap='jet')
      for x in range(1,num_x+1):
        plt.axvline(x=x*region_size,color='r')
      for y in range(1,num_y+1):
        plt.axhline(y=y*region_size,color='r')    
      plt.colorbar()
    vsrc_nodes = load_vsrc_file(cur_design, data_dir)
    macro_areas = load_macros(cur_design, data_dir)
    signal_congestion = get_congestion(current_map, cur_design, 
                                       data_dir, num_x, num_y) 
    logger_h.debug("processing_maps")                                      
    processed_maps = process_maps( current_map, signal_congestion,
                                   vsrc_nodes, macro_areas,
                                   num_x, num_y)
    logger_h.debug("processed maps")                                      
    cong_map, macro_map, eff_dist_map = processed_maps
    regions=[]
    for x in range(num_x):
      for y in range(num_y):
        all_maps = np.zeros((4,) + (current_map.shape[-2:]))
        all_maps[0,:,:] = current_map
        all_maps[1,:,:] = cong_map
        all_maps[2,:,:] = macro_map
        all_maps[3,:,:] = eff_dist_map
        op_data = extract_region(all_maps,region_size,x,y)
        current_maps = np.concatenate((current_maps,op_data[0:1,...]))
        cong_maps = np.concatenate((cong_maps,op_data[1:2,...]))
        macro_maps = np.concatenate((macro_maps,op_data[2:3,...]))
        eff_dist_maps = np.concatenate((eff_dist_maps,op_data[3:4,...]))
        n = x*num_y + y
        lx, ux = int(x*region_size), int(min((x+1)*region_size,x_dim))
        ly, uy = int(y*region_size), int(min((y+1)*region_size,y_dim))
        lx,ux,ly,uy = ( lx* grid_params['unit_micron'],
                        ux* grid_params['unit_micron'],
                        ly* grid_params['unit_micron'],
                        uy * grid_params['unit_micron'])
        regions.append(((lx,ux),(ly,uy)))

    inf_data[cur_design] = (current_maps, cong_maps, 
                            macro_maps, eff_dist_maps)
  return inf_data

def process_maps(
    current_maps, 
    signal_congestion, 
    vsrc_maps,
    blockage_maps,
    num_x, num_y):
  cong_maps = np.ones(current_maps.shape)
  macro_maps = np.zeros(current_maps.shape)
  eff_dist_maps =  region_size*np.ones(current_maps.shape)
  x_dim = current_maps.shape[-2]
  y_dim = current_maps.shape[-1]
  for x in range(num_x):
    for y in range(num_y):
      n = x*num_y + y
      lx = (x  )*region_size
      ux = (x+1)*region_size
      ly = (y  )*region_size
      uy = (y+1)*region_size
      cong_maps[lx:ux,ly:uy] = signal_congestion[n]
  blockages = blockage_maps
  if blockages is not None:
    for blockage in blockages:
      ux,lx = (int(blockage[1][0][1]/grid_params['unit_micron']), 
               int(blockage[1][0][0]/grid_params['unit_micron']))
      uy,ly = (int(blockage[1][1][1]/grid_params['unit_micron']),
               int(blockage[1][1][0]/grid_params['unit_micron']))
      macro_maps[lx:ux,ly:uy] = 1
  vsrc_nodes = np.array(vsrc_maps,ndmin=2)
  pbar = tqdm(total = x_dim*y_dim)
  for x in range(x_dim):
    for y in range(y_dim):
      d_inv = 0
      for vsrc_node in vsrc_nodes:
        x_loc = vsrc_node[0]/grid_params['unit_micron']
        y_loc = vsrc_node[1]/grid_params['unit_micron']
        d = np.sqrt((x-x_loc)**2 + (y-y_loc)**2)
        if d == 0:
          d_inv = -1
          break
        else:
          d_inv = d_inv+ 1/d
      if d_inv <0 :
        eff_dist_maps[x,y] = 0
      else:
        eff_dist_maps[x,y] = 1/d_inv
      pbar.update(1)
  return cong_maps, macro_maps, eff_dist_maps

def load_vsrc_file(design, data_dir):
  fname = "%s/%s/vsrc.txt"%(data_dir,design)
  logger_h.debug("Loading VSRC File: %s"%fname)
  arr =  np.loadtxt(fname,
        delimiter=',')
  return np.array(arr, ndmin=2)
  
def load_macros(design, data_dir):
  fname = "%s/%s/macro.txt"%(data_dir,design)
  logger_h.debug("Loading Macros file: %s"%fname)
  macros =[]
  with open(fname,'r') as f:
    for line in f:
      layers_str, values_str = line.split(';')
      macro_layers = [int(x) for x in layers_str.split()]
      x, urx, y, ury = [int(x) for x in values_str.split()]
      macros.append((macro_layers,((x,urx),(y,ury))))

  return macros

def get_congestion(current_map, design, data_dir, num_x, num_y):
  cong_maps = load_congestion(design, data_dir)
  signal_congestion = np.zeros(num_x*num_y)
  for x in range(num_x):
    for y in range(num_y):
      lx, ux = x*region_size, (x+1)*region_size
      ly, uy = y*region_size, (y+1)*region_size
      region = cong_maps[lx:ux,ly:uy]
      signal_congestion[x*num_y+y]  = np.mean(region)
  return signal_congestion


def load_congestion(design, data_dir):
  fname = "%s/%s/congestion.csv"%(data_dir,design)
  logger_h.debug("Loading congestion file :%s"%fname)
  return np.loadtxt(fname,
        delimiter=',')


