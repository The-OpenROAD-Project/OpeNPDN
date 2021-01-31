from ir_solver import ir_solver
import numpy as np
import matplotlib.pyplot as plt
from current_mapgen import current_mapgen
import json
import logging

def template_elimination(template_file,
      grid_params_file,
      ir_params_file,
      log_level,
      logger_file=None):
  global logger_h
  logger_h = create_logger(log_level, logger_file)
  logger_h.info("Starting template elimination script")
  set_globals(grid_params_file, ir_params_file)
  logger_h.info("Loaded global parameters:")
  logger_h.info("  Grid parameters: %s"%grid_params_file)
  logger_h.info("  IR solver parameters: %s"%ir_params_file)
  logger_h.debug("Loaded json value:%d"%grid_params["unit_micron"])
  vsrc_nodes = generate_vsrc()
  logger_h.debug("Generated vsrc nodes")
  logger_h.debug("%s"%vsrc_nodes)
  curr_mapgen_h = current_mapgen(grid_params)
  current_maps_dok = curr_mapgen_h.to_dok_map(current_map)
  
  logger_h.debug("Initializing the IR solver")
  ir_solver_h = ir_solver(ir_solver_params,grid_params)
  ir_solver_h.build(current_maps_dok[0],vsrc_nodes)  
  logger_h.debug("IR solver initialized")
  
  plot_selected = logger_h.getEffectiveLevel() <= logging.DEBUG
  selected_templates = eliminate_templates(ir_solver_h, templates, plot=plot_selected)
  logger_h.info("Saving template information to : %s"%template_file)
  
  with open(template_file, 'w') as f:
    for n in selected_templates:
      templ = templates[n]
      for t, layer in enumerate(templ):
        f.write("%2d, %7d"%layer)
        if t != len(templ)-1:
          f.write(", ")
      f.write("\n")

def set_globals(grid_params_file, ir_params_file):
  global ir_solver_params
  global grid_params
  global current_map
  global x_dim
  global y_dim
  global templates
  with open(grid_params_file) as f:
    grid_params= json.load(f)
  with open(ir_params_file) as f:
    ir_solver_params = json.load(f)
  x_dim = 251
  y_dim = 251
  max_cur = 5e-7
  current_map = np.ones((x_dim, y_dim)) * max_cur 
  grid_params["size_x"] = x_dim*grid_params["unit_micron"]
  grid_params["size_y"] = y_dim*grid_params["unit_micron"]
  templates = [] 
  for l3 in [1,2,4]: 
    for l2 in [1,2.5,5]: 
      for l1 in [1,3,6]: 
        templates.append([(1,l1*40000),(2,l2*40000),(3,l3*40000)])
  
def generate_vsrc():
  vsrc_pitch  = 200
  vsrc_offset_x = 125
  vsrc_offset_y = 125
  
  num_bump_x = int((x_dim - vsrc_offset_x) /vsrc_pitch)+1
  num_bump_y = int((y_dim - vsrc_offset_y) /vsrc_pitch)+1
  num_bumps = num_bump_x*num_bump_y
  
  vsrc_nodes = []
  vdd = ir_solver_params["vdd"]
  for vsrc_node_loc in range(num_bumps):
    lx = int(vsrc_node_loc/num_bump_y) 
    ly = vsrc_node_loc % num_bump_y
    llx = (vsrc_offset_x + lx*vsrc_pitch)*grid_params["unit_micron"]
    lly = (vsrc_offset_y + ly*vsrc_pitch)*grid_params["unit_micron"]
    vsrc_nodes.append([llx,lly,vdd])
  return vsrc_nodes

def pdn_util(template_num,grid_params,templates):
  t_spacing = [4800, 4800, 8000, 8000, 8000]
  num_stripes = [grid_params['size_x']/4800,\
                 grid_params['size_y']/4800,\
                 grid_params['size_x']/8000,\
                 grid_params['size_y']/8000,\
                 grid_params['size_x']/8000]
  template = templates[template_num]
  util=0
  for layer, pitch in template:
    util += num_stripes[layer] * t_spacing[layer]/pitch
  total_util = util/sum(num_stripes)   
  return total_util

def eliminate_templates(ir_solver_h, templates, plot=False):
  ir_drops = np.zeros(len(templates))
  temp_util = np.zeros(len(templates))
  new_grid = ir_solver_h.internal_grid
  for n in range(len(templates)):
      logger_h.debug("Running template %d"%n)
      layers_info = templates[n]
      new_grid.update_region((0,grid_params['size_x']),(0,grid_params['size_y']),layers_info)
      logger_h.debug("Updated grid %d"%n)
      _,ir_drops[n],_ = ir_solver_h.solve_ir()
      logger_h.debug("WIR %f"%ir_drops[n])
      temp_util[n] = pdn_util(n,grid_params,templates)
      logger_h.debug("Template_utilization %f"%temp_util[n])
  logger_h.debug("ir drops :\n%s\nutilization:\n%s"%(ir_drops, temp_util))
  
  selected_templates = []
  eps = 1.95e-3
  eps_c = 1e-3
  for n in range(len(templates)):
      temp_selected = True
      for i in range(len(templates)):
          if (ir_drops[n]> ir_drops[i] - eps and temp_util[n] > temp_util[i] and i != n):
              temp_selected= False
              break
          elif(abs(temp_util[n]-temp_util[i])<eps_c and ir_drops[n]>ir_drops[i] and i != n):
              temp_selected= False
              break            
      if temp_selected:
          selected_templates.append(n)
  ind = np.argsort(ir_drops[selected_templates])
  selected_templates = np.array(selected_templates)
  logger_h.debug("Selected templates: %s"%selected_templates[ind])
  if plot:
    plot_templates(ir_drops, temp_util, selected_templates[ind])
  return selected_templates[ind]

def plot_templates(ir_drops, utils, selected_templates=None):
  for n in range(len(templates)):
    plt.scatter(ir_drops[n],utils[n])
    if (selected_templates is not None) and (n in selected_templates):
      plt.text(ir_drops[n]+0.0002, utils[n]+0.001, str(n));
    if selected_templates is not None:
      plt.plot(ir_drops[selected_templates], utils[selected_templates])
  plt.show()

def create_logger(log_level, log_file=None):
  # Create a custom logger
  logger = logging.getLogger("TPEL")
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

if __name__ == "__main__":
  #log_level = logging.DEBUG
  log_level = logging.INFO
  template_file = "./params/templates.csv"
  grid_params_file = "./params/grid_params.json"
  ir_params_file = "./params/IR_params.json"
  template_elimination(template_file,
      grid_params_file,
      ir_params_file,
      log_level)
