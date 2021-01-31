import json
import logging
import numpy as np
from current_mapgen import current_mapgen
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from ir_solver import ir_solver
from simulated_annealing import simulated_annealer

def sythetic_data_gen(grid_params_file,
      ir_params_file,
      template_file,
      log_level,
      data_dir,
      logger_file=None):
  global logger_h
  logger_h = create_logger(log_level, logger_file)
  plot_maps = logger_h.getEffectiveLevel()<=logging.DEBUG

  logger_h.info("Starting synthetic data generation.")
  set_globals(grid_params_file, ir_params_file)
  logger_h.info("Loaded global parameters:")
  logger_h.info("  Grid parameters: %s"%grid_params_file)
  logger_h.info("  IR solver parameters: %s"%ir_params_file)
  logger_h.info("  Templates: %s"%template_file)
  logger_h.debug("Loaded json value:%d"%grid_params['size_x'])
  templates = load_templates(template_file)
  sa_params = set_sa_params()
  logger_h.info("Generating the input data.")
  current_maps, current_maps_dok, signal_congestion = generate_maps(plot= plot_maps)
  vsrc_maps = generate_vsrc()
  blockage_maps = generate_marcos()
  cong_maps, macro_maps, eff_dist_maps = process_maps(current_maps,
                                            signal_congestion, vsrc_maps,
                                            blockage_maps, plot_maps)
  
  regions = []
  for x in range(NUM_REGIONS_X):
    for y in range(NUM_REGIONS_Y):
        n = x*NUM_REGIONS_Y + y
        lx, ux = x*region_size, (x+1)*region_size
        ly, uy = y*region_size, (y+1)*region_size
        lx,ux,ly,uy = lx* grid_params['unit_micron'],ux* grid_params['unit_micron'],ly* grid_params['unit_micron'],uy * grid_params['unit_micron']
        regions.append(((lx,ux),(ly,uy)))

  state = np.zeros((NUM_MAPS,NUM_REGIONS),dtype='int')
  logger_h.info("Running simulated annealing.")
  pbar = tqdm(current_maps_dok, position=0)
  for map_num, current_map in enumerate(pbar):
    ir_solver_h = ir_solver(ir_solver_params,grid_params)
    ir_solver_h.build(current_map,vsrc_maps[map_num], blockages=blockage_maps[map_num])
    init_state = np.zeros(NUM_REGIONS,dtype='int')
    pdn_opt = simulated_annealer(
                        init_state, sa_params['T_init'], 
                        sa_params['T_final'], sa_params['alpha_temp'], 
                        sa_params['num_moves_per_step'], 
                        signal_congestion[map_num], ir_solver_h,
                        templates,regions)
    state[map_num],_ = pdn_opt.sim_anneal(plot_maps)

  logger_h.info("Simulated annealing completed. Storing the result.")
  store_maps(current_maps, cong_maps, macro_maps, 
                eff_dist_maps, state, data_dir, plot_maps)


  if plot_maps:
    plt.show()
  
def store_maps(current_maps, cong_maps, macro_maps, 
                eff_dist_maps, state, data_dir, plot =False):
  op_current_maps = np.zeros((NUM_REGIONS, 3*region_size, 3*region_size))
  op_congestion_maps = np.ones((NUM_REGIONS,3*region_size,3*region_size))
  op_macro_maps  = np.zeros((NUM_REGIONS,3*region_size,3*region_size))
  op_eff_dist_maps = region_size*np.ones((NUM_REGIONS,3*region_size,3*region_size))
          
  for map_num in trange(NUM_MAPS):
    for x in range(NUM_REGIONS_X):
      for y in range(NUM_REGIONS_Y):
        n = x*NUM_REGIONS_Y + y
        lx = max((x-1)*region_size, 0)
        ux = min((x+2)*region_size, NUM_REGIONS_X*region_size)
        ly = max((y-1)*region_size, 0)
        uy = min((y+2)*region_size, NUM_REGIONS_Y*region_size)
        region_curr = current_maps[map_num,lx:ux,ly:uy]
        region_cong = cong_maps[map_num,lx:ux,ly:uy]
        region_macro = macro_maps[map_num,lx:ux,ly:uy]
        region_eff_d = eff_dist_maps[map_num,lx:ux,ly:uy]
        if x == 0 :
            rlx = region_size
        else:
            rlx = 0
        if y == 0 :
            rly = region_size
        else:
            rly = 0
        if x == NUM_REGIONS_X-1:
            rux = 2*region_size
        else:    
            rux = 3*region_size
        if y == NUM_REGIONS_Y-1:
            ruy = 2*region_size
        else:    
            ruy = 3*region_size
        op_current_maps[n,rlx:rux,rly:ruy] = region_curr
        op_congestion_maps[n,rlx:rux,rly:ruy] = region_cong
        op_macro_maps[n,rlx:rux,rly:ruy] = region_macro
        op_eff_dist_maps[n,rlx:rux,rly:ruy] = region_eff_d
        if map_num ==1 and n == 1 and plot:
          fig,ax = plt.subplots(1,4)
          ax[0].imshow(op_current_maps[n].T,origin='lower')
          ax[1].imshow(op_congestion_maps[n].T,origin='lower')
          ax[2].imshow(op_macro_maps[n].T,origin='lower')
          ax[3].imshow(op_eff_dist_maps[n].T,origin='lower')

    np.savez("%s/synth_data_%d.npz"%(data_dir, map_num+10),
           current_maps    = op_current_maps,
           congestion_maps = op_congestion_maps,
           macro_maps      = op_macro_maps,
           eff_dist_maps   = op_eff_dist_maps,
           state           = state[map_num]
          )
    logger_h.debug(state[map_num])
    #logger_h.debug(op_current_maps.shape)
    #logger_h.debug(op_congestion_maps.shape)
    #logger_h.debug(op_macro_maps.shape)
    #logger_h.debug(op_eff_dist_maps.shape)


def process_maps(
    current_maps, 
    signal_congestion, 
    vsrc_maps,
    blockage_maps,
    plot=False):
  cong_maps = np.ones(current_maps.shape)
  macro_maps = np.zeros(current_maps.shape)
  eff_dist_maps =  region_size*np.ones(current_maps.shape)
  for map_num in range(current_maps.shape[0]):
    for x in range(NUM_REGIONS_X):
      for y in range(NUM_REGIONS_Y):
        n = x*NUM_REGIONS_Y + y
        lx = (x  )*region_size
        ux = (x+1)*region_size
        ly = (y  )*region_size
        uy = (y+1)*region_size
        cong_maps[map_num,lx:ux,ly:uy] = signal_congestion[map_num,n]
    blockages = blockage_maps[map_num]
    if blockages is not None:
      for blockage in blockages:
        ux,lx = int(blockage[1][0][1]/grid_params['unit_micron']), int(blockage[1][0][0]/grid_params['unit_micron'])
        uy,ly = int(blockage[1][1][1]/grid_params['unit_micron']), int(blockage[1][1][0]/grid_params['unit_micron'])
        macro_maps[map_num,lx:ux,ly:uy] = 1
    vsrc_nodes = vsrc_maps[map_num]
    for x in range(random_map_params['x_dim']):
      for y in range(random_map_params['y_dim']):
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
          eff_dist_maps[map_num,x,y] = 0
        else:
          eff_dist_maps[map_num,x,y] = 1/d_inv
  if plot:
    plt.figure()
    plt.imshow(current_maps[0].T,origin='lower',cmap='jet')
    plt.colorbar()
    plt.figure()
    plt.imshow(cong_maps[0].T,origin='lower',cmap='jet')
    plt.colorbar()
    plt.figure()
    plt.imshow(macro_maps[0].T,origin='lower',cmap='jet')
    plt.colorbar()
    plt.figure()
    plt.imshow(eff_dist_maps[0].T,origin='lower',cmap='jet')
    plt.colorbar()
  return cong_maps, macro_maps, eff_dist_maps

def generate_marcos():
  logger_h.debug("Generating Macros")
  rng = np.random.default_rng()
  macro_layers = [0,1,2]
  blockage_maps = []
  for map_num in trange(NUM_MAPS):
    blockages = []
    num_blockages = rng.integers(10)
    #num_blockages = 1

    while(len(blockages)<num_blockages):
      blocked = False
      x =  np.random.randint(random_map_params['x_dim'])
      y =  np.random.randint(random_map_params['y_dim'])
      width = np.random.randint(10,50)
      aspect = 2 + np.random.random()*2
      if np.random.randint(2) == 1:
        dx = width
        dy = int(width*aspect)
      else:
        dx = int(width*aspect)
        dy = width
      if( x<40 or y<10 or 0 < (random_map_params['x_dim'] - (x+dx)) < 40  
          or 0 < (random_map_params['y_dim'] - (y+dy)) <20):
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
      blockages_scaled = None
    else:
      blockages_scaled = []
      for layer,region in blockages:
        xval, yval = region
        x = xval[0] * grid_params['unit_micron']
        urx = xval[1] * grid_params['unit_micron']
        y = yval[0] * grid_params['unit_micron']
        ury = yval[1] * grid_params['unit_micron']
        blockages_scaled.append((macro_layers,((x,urx),(y,ury))))

    blockage_maps.append(blockages_scaled)
  return blockage_maps
  
def generate_vsrc():
  logger_h.debug("Generating Vsrc")
  rng = np.random.default_rng()
  vsrc_maps = []
  for _ in trange(NUM_MAPS):
    vsrc_pitch  = rng.integers(100,150) 
    vsrc_offset_x = rng.integers(0,vsrc_pitch) 
    vsrc_offset_y = rng.integers(0,vsrc_pitch) 

    num_bump_x = int((random_map_params['x_dim'] - vsrc_offset_x) /vsrc_pitch)+1
    num_bump_y = int((random_map_params['y_dim'] - vsrc_offset_y) /vsrc_pitch)+1
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
    vsrc_maps.append(vsrc_nodes)
  return vsrc_maps

def generate_maps(plot=False):
  logger_h.debug("Generating current and congestion maps")
  curr_mapgen_h = current_mapgen(grid_params)
  current_maps = curr_mapgen_h.gen_random_maps(output_str=None,
                                             random_map_params=random_map_params)
  cong_maps_rand = curr_mapgen_h.gen_random_maps(random_map_params=random_cong_map_params)
  cong_maps = cong_maps_rand *current_maps
  cong_max = np.max(cong_maps)
  cong_maps = cong_maps/cong_max
  if plot:
    plt.figure()
    plt.imshow(current_maps[0].T,cmap='jet',origin='lower')
    plt.colorbar()
    plt.figure()
    plt.imshow(cong_maps[0].T,cmap='jet',origin='lower')
    plt.colorbar()
  signal_congestion = np.zeros((NUM_MAPS, NUM_REGIONS))
  for n,cong_map in enumerate(cong_maps):
    for x in range(NUM_REGIONS_X):
      for y in range(NUM_REGIONS_Y):
        lx, ux = x*region_size, (x+1)*region_size
        ly, uy = y*region_size, (y+1)*region_size
        region = cong_map[lx:ux,ly:uy]
        signal_congestion[n,x*NUM_REGIONS_Y+y] = np.mean(region)
  current_maps_dok = curr_mapgen_h.to_dok_map(current_maps)
  return current_maps, current_maps_dok, signal_congestion

def load_templates(template_file):
  logger_h.debug("Loading templates")
  templates = np.loadtxt(template_file,delimiter= ',',dtype=np.int)
  num_temp, _ = templates.shape
  templates = templates.reshape((num_temp,-1,2))
  logger_h.debug("Templates:\n%s"%templates.reshape((num_temp,-1)))
  return templates

def set_sa_params():
  sa_params = {}
  sa_params['T_init'] = 70
  sa_params['T_final'] = 1e-4
  sa_params['alpha_temp'] = 0.95
  sa_params['num_moves_per_step'] = 5

  return sa_params

def create_logger(log_level, log_file=None):
  # Create a custom logger
  logger = logging.getLogger("SYDG")
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


def set_globals(grid_params_file, ir_params_file):
  global ir_solver_params
  global random_map_params
  global grid_params
  global random_cong_map_params
  global NUM_MAPS
  global region_size
  global NUM_REGIONS_X, NUM_REGIONS_Y, NUM_REGIONS

  with open(grid_params_file) as f:
    grid_params= json.load(f)
  with open(ir_params_file) as f:
    ir_solver_params = json.load(f)

  random_map_params = {
    'x_dim' : 501,
    'y_dim' : 501,
    'var' : 80,
    'max_cur' : 2.5e-7,
    'len_scale' : 200,
    'num_maps' : 20
  }
  grid_params['size_x'] = random_map_params['x_dim']*grid_params['unit_micron']
  grid_params['size_y'] = random_map_params['y_dim']*grid_params['unit_micron']
  random_cong_map_params = dict(random_map_params)
  random_cong_map_params['len_scale'] = 40
  random_cong_map_params['max_cur'] = 1
  
  region_size =100
  NUM_MAPS = random_map_params['num_maps']
  NUM_REGIONS_X = int(random_map_params['x_dim']/region_size)
  NUM_REGIONS_Y = int(random_map_params['y_dim']/region_size)
  NUM_REGIONS = NUM_REGIONS_X * NUM_REGIONS_Y

if __name__ == '__main__':
  #log_level = logging.DEBUG
  log_level = logging.INFO
  grid_params_file = "./params/grid_params.json"
  ir_params_file = "./params/IR_params.json"
  template_file = "./params/templates.csv"
  data_dir = "./run/data"
  sythetic_data_gen(
      grid_params_file,
      ir_params_file,
      template_file,
      log_level,
      data_dir,
      logger_file=None)
