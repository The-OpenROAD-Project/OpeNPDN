import json
import logging
import numpy as np
from current_mapgen import current_mapgen
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from ir_solver import ir_solver
from simulated_annealing import simulated_annealer
import scipy
import os

def TL_data_gen(grid_params_file,
      ir_params_file,
      template_file,
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
  templates = load_templates(template_file)
  sa_params = set_sa_params()
  logger_h.info("Generating the input data.")
  #'bp' 32
  designs =\
  ['aes','ibex','jpeg','dynamic_node','bp_fe','bp_be','bp_multi','swerv']
  cur_scale =\
  [   15,    10,    10,            15,     30,     35,        15,   12.5]
  save_data, sa_data = load_data('./designs', designs, cur_scale,plot_maps)
  #current_maps, current_maps_dok, signal_congestion = generate_maps(plot= plot_maps)
  
  logger_h.info("Running simulated annealing.")
  pbar = tqdm(sa_data, position=0)
  for map_num, sa_pt in enumerate(pbar):
    (current_map, vsrc_maps, blockage_maps, 
      signal_congestion, regions, size_designs) = sa_pt

    grid_params['size_x'] = current_map.shape[-2]*grid_params['unit_micron']
    grid_params['size_y'] = current_map.shape[-1]*grid_params['unit_micron']
    ir_solver_h = ir_solver(ir_solver_params,grid_params)
    curr_mapgen_h = current_mapgen(grid_params) 
    current_map_dok = curr_mapgen_h.to_dok_map(current_map)
    ir_solver_h.build(current_map_dok[0],
                      vsrc_maps,
                      blockages=blockage_maps)
    init_state = np.zeros(len(regions),dtype='int')
    logger_h.debug("Building Simulated annealing on: %s"%designs[map_num])
    pdn_opt = simulated_annealer(
                        init_state, sa_params['T_init'], 
                        sa_params['T_final'], sa_params['alpha_temp'], 
                        sa_params['num_moves_per_step'], 
                        signal_congestion, ir_solver_h,
                        templates,regions)
    logger_h.debug("Running Simulated annealing on: %s"%designs[map_num])
    op_state,_ = pdn_opt.sim_anneal(plot_maps)
    store_maps(save_data[map_num], op_state, data_dir, designs[map_num])

  logger_h.info("Simulated annealing completed. Storing the result.")


  if plot_maps:
    plt.show()

def store_maps(save_data, state, data_dir, design):
  current_maps, cong_maps, macro_maps, eff_dist_maps = save_data
  direc = "%s/%s"%(data_dir, design)
  if not os.path.exists(direc):
    os.makedirs(direc)
  np.savez("%s/TL_data.npz"%(direc),
           current_maps    = current_maps,
           congestion_maps = cong_maps,
           macro_maps      = macro_maps,
           eff_dist_maps   = eff_dist_maps,
           state           = state
          )
  logger_h.debug("%s : %s"%(design,state))

  

def load_data(data_dir, designs, cur_scale, plot):
  data_stats = {}
  start,end = 0,0
  
  save_data=[]
  sa_data=[]

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
    data_stats[cur_design] ={}
    data_stats[cur_design]['start'], data_stats[cur_design]['end'] = start,end
    data_stats[cur_design]['nx'],data_stats[cur_design]['ny'] = num_x,num_y

    if plot and n==0:
      plt.figure()
      plt.imshow(current_map.T,cmap='jet')
      for x in range(1,num_x+1):
        plt.axvline(x=x*region_size,color='r')
      for y in range(1,num_y+1):
        plt.axhline(y=y*region_size,color='r')    
      plt.colorbar()
    
    #vsrc_nodes = generate_vsrc_file(current_map, cur_design, data_dir)
    vsrc_nodes = load_vsrc_file(cur_design, data_dir)
    #macro_areas = generate_macros(current_map, cur_design, data_dir)
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

    save_data.append((current_maps, cong_maps, macro_maps, eff_dist_maps))

    sa_data.append((current_map, vsrc_nodes, macro_areas, 
              signal_congestion, regions, (num_x,num_y)))

  return save_data, sa_data

def get_congestion(current_map, design, data_dir, num_x, num_y):
  #cong_maps = generate_congestion(current_map, design, data_dir)
  cong_maps = load_congestion(design, data_dir)
  signal_congestion = np.zeros(num_x*num_y)
  for x in range(num_x):
    for y in range(num_y):
      lx, ux = x*region_size, (x+1)*region_size
      ly, uy = y*region_size, (y+1)*region_size
      region = cong_maps[lx:ux,ly:uy]
      signal_congestion[x*num_y+y]  = np.mean(region)
  return signal_congestion

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


def load_congestion(design, data_dir):
  fname = "%s/%s/congestion.csv"%(data_dir,design)
  logger_h.debug("Loading congestion file :%s"%fname)
  return np.loadtxt(fname,
        delimiter=',')

def generate_congestion(current_map, design, data_dir):
  logger_h.debug("Generating congestion maps")
  x_dim = current_map.shape[-2]
  y_dim = current_map.shape[-1]
  random_map_params = {
    'x_dim' : x_dim,
    'y_dim' : y_dim,
    'var' : 80,
    'max_cur' : 1,
    'len_scale' : 40,
    'num_maps' : 1  }
  curr_mapgen_h = current_mapgen(grid_params) 
  cong_maps_rand = curr_mapgen_h.gen_random_maps(random_map_params=random_map_params)
  cong_maps = cong_maps_rand[0] * current_map
  cong_max = np.max(cong_maps)
  cong_maps = cong_maps/cong_max
  np.savetxt("%s/%s/congestion.csv"%(data_dir,design),
        cong_maps,
        delimiter=',')

  return cong_maps


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

def generate_macros(current_map,design, data_dir):
  logger_h.debug("Generating Macros")
  rng = np.random.default_rng()
  macro_layers = [0,1,2]
  blockage_maps = []
  blockages = []
  num_blockages = rng.integers(1,10)
  x_dim = current_map.shape[0]
  y_dim = current_map.shape[1]

  while(len(blockages)<num_blockages):
    blocked = False
    x =  np.random.randint(x_dim)
    y =  np.random.randint(y_dim)
    width = np.random.randint(10,50)
    aspect = 2 + np.random.random()*2
    if np.random.randint(2) == 1:
      dx = width
      dy = int(width*aspect)
    else:
      dx = int(width*aspect)
      dy = width
    if( x<40 or y<10 or 0 < (x_dim - (x+dx)) < 40  
        or 0 < (y_dim - (y+dy)) <20):
      continue
    for blockage in blockages:
      (blx,bux),(bly,buy) = blockage[1]
      #creates vertical channel with prior blockage
      if (0 < blx - (x+dx) < 40 ) or (0 < x - bux < 40 ):
        if not (bly >= (y+dy) or  buy <= y):
          blocked = True
          break
      #creates horizontal channel with prior blockage
      if (0 < bly - (y+dy) < 20 ) or (0 < y - buy < 20 ):
        if not (blx >= (x+dx) or  bux <= x):
          blocked = True
          break

    if not blocked:
      blockages.append((macro_layers,((x,x+dx),(y,y+dy))))

  if num_blockages == 0:
    blockage_maps= None
  else:
    blockage_maps= []
    for layer,region in blockages:
      xval, yval = region
      x = xval[0] * grid_params['unit_micron']
      urx = xval[1] * grid_params['unit_micron']
      y = yval[0] * grid_params['unit_micron']
      ury = yval[1] * grid_params['unit_micron']
      blockage_maps.append((macro_layers,((x,urx),(y,ury))))
  with open("%s/%s/macro.txt"%(data_dir,design),'w') as f:
    for blockage in blockage_maps:
      f.write("%s;"%(" ".join([str(x) for x in blockage[0]])))
      coords = blockage[1][0] + blockage[1][1]
      f.write("%s\n"%(" ".join([str(x) for x in coords])))
      

  return blockage_maps


def load_vsrc_file(design, data_dir):
  fname = "%s/%s/vsrc.txt"%(data_dir,design)
  logger_h.debug("Loading VSRC File: %s"%fname)
  arr =  np.loadtxt(fname,
        delimiter=',')
  return np.array(arr, ndmin=2)


def generate_vsrc_file(cur_map, design, data_dir):
  logger_h.debug("Generating_vsrc")
  rng = np.random.default_rng()
  vsrc_pitch  = rng.integers(100,150) 
  vsrc_offset_x = rng.integers(0,vsrc_pitch) 
  vsrc_offset_y = rng.integers(0,vsrc_pitch) 
  x_dim = cur_map.shape[-2]
  y_dim = cur_map.shape[-1]
  num_bump_x = int((x_dim - vsrc_offset_x) /vsrc_pitch)+1
  num_bump_y = int((y_dim - vsrc_offset_y) /vsrc_pitch)+1
  num_bumps = num_bump_x*num_bump_y

  bump_mat = np.array([[x+y*num_bump_x for x in range(num_bump_x)]for y in range(num_bump_y)])
  bump_list = []
  for x in range(0,num_bump_x,2):
    for y in range(0,num_bump_y,2):
      bump_list.append(bump_mat[y:y+2,x:x+2,].reshape(-1))
  bump_list = np.array(bump_list)
  vsrc_node_locs = []
  for row in bump_list:
    vsrc_node_locs.append(rng.choice(row))

  vsrc_nodes = []
  for vsrc_node_loc in vsrc_node_locs:
    lx = int(vsrc_node_loc/num_bump_y) 
    ly = vsrc_node_loc % num_bump_y
    llx = (vsrc_offset_x + lx*vsrc_pitch)*grid_params['unit_micron']
    lly = (vsrc_offset_y + ly*vsrc_pitch)*grid_params['unit_micron']
    vsrc_nodes.append([llx,lly,ir_solver_params['vdd']])
  np.savetxt("%s/%s/vsrc.txt"%(data_dir,design),
        vsrc_nodes,
        delimiter=',')
  return vsrc_nodes

def extract_region(all_maps, region_size,x,y):    
  op_maps = np.zeros((all_maps.shape[0],3*region_size,3*region_size))
  lx = max((x-1)*region_size, 0)
  rlx = max(0, lx - (x-1)*region_size)
  ux = min((x+2)*region_size, all_maps.shape[-2])
  rux = region_size + min( 2*region_size, 2*region_size + ux-(x+2)*region_size )
  ly = max((y-1)*region_size, 0)
  rly = max(0, ly - (y-1)*region_size)
  uy = min((y+2)*region_size, all_maps.shape[-1])
  ruy = region_size + min( 2*region_size, 2*region_size + uy-(y+2)*region_size)
  op_maps[:,rlx:rux,rly:ruy] = all_maps[:,lx:ux,ly:uy]
      
  return op_maps

def create_logger(log_level, log_file=None):
  # Create a custom logger
  logger = logging.getLogger("TLDG")
  logger.setLevel(log_level)
  
  c_handler = logging.StreamHandler()
  c_handler.setLevel(log_level)
  
  # Create formatters and add it to handlers
  c_format = logging.Formatter('[%(name)s][%(levelname)s][%(message)s]')
  c_handler.setFormatter(c_format)
  
  # Add handlers to the logger
  logger.addHandler(c_handler)

  # Process only if log file defined
  if log_file is not None:
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.WARNING)
    f_format = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(message)s]')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
  return logger
def set_sa_params():
  sa_params = {}
  sa_params['T_init'] = 70
  sa_params['T_final'] = 1e-4
  sa_params['alpha_temp'] = 0.95
  sa_params['num_moves_per_step'] = 5

  return sa_params

def set_globals(grid_params_file, ir_params_file):
  global ir_solver_params
  global random_map_params
  global grid_params
  global random_cong_map_params
  global region_size

  with open(grid_params_file) as f:
    grid_params= json.load(f)
  with open(ir_params_file) as f:
    ir_solver_params = json.load(f)

#    'len_scale' : 200,
#    'num_maps' : 20
#  }
#  random_cong_map_params = dict(random_map_params)
#  random_cong_map_params['len_scale'] = 40
#  random_cong_map_params['max_cur'] = 1
  
  region_size =100

def load_templates(template_file):
  logger_h.debug("Loading templates")
  templates = np.loadtxt(template_file,delimiter= ',',dtype=np.int)
  num_temp, _ = templates.shape
  templates = templates.reshape((num_temp,-1,2))
  logger_h.debug("Templates:\n%s"%templates.reshape((num_temp,-1)))
  return templates

if __name__ == '__main__':
  log_level = logging.DEBUG
  #log_level = logging.INFO
  grid_params_file = "./params/grid_params.json"
  ir_params_file = "./params/IR_params.json"
  template_file = "./params/templates.csv"
  data_dir = "./run/TL_data"
  TL_data_gen(
      grid_params_file,
      ir_params_file,
      template_file,
      log_level,
      data_dir,
      logger_file=None)

