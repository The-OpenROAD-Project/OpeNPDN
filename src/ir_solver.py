import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sparse_algebra
from grid import grid
from pprint import pprint
from time import time
from tqdm import tqdm

import sys
from os.path import dirname
sys.path.append(dirname(__file__))
from current_mapgen import current_mapgen
import matplotlib.pyplot as plt

class ir_solver:
  def __init__(self, ir_solver_params, grid_params):
    self.grid_params = grid_params
    self.grid = grid(grid_params)
    self.vdd = ir_solver_params['vdd']
    self.ir_drop_limit = ir_solver_params['ir_drop_limit']
  
  @property 
  def internal_grid(self):
    return self.grid

  def set_grid(self, grid):
    self.grid = grid
    self._G = grid.Gmat

  def generate_Vsrc_nodes(self,vsrc_nodes):
    num_vsrc = len(vsrc_nodes)
    Gmat = self.grid.Gmat 
    num_nodes = self.grid.num_nodes
    size = num_nodes + num_vsrc
    Gmat.resize((size,size))
    top_layer = self.grid.top_layer
    for vsrc_num,vsrc_node in enumerate(vsrc_nodes):
      x = vsrc_node[0]
      y = vsrc_node[1]
      node_h = self.grid.get_node(top_layer, x, y, True)  
      Gmat[node_h.G_loc, num_nodes+vsrc_num] = 1
      Gmat[num_nodes+vsrc_num, node_h.G_loc] = 1

  def create_J(self, current_map, vsrc_nodes, blockages=None):
    blockages_present = False
    if blockages is not None:
      for layers, (x_range,y_range) in blockages:
        if self.grid.bottom_layer in layers:
            blockages_present = True
            
    self._J_nodes= []
    num_vsrc = len(vsrc_nodes)
    num_nodes = self.grid.num_nodes
    size = num_nodes + num_vsrc
    bottom_layer = self.grid.bottom_layer
    J = np.zeros((size,1))
    for current_node in current_map:
      x = current_node[0]
      y = current_node[1]
      I = current_node[2]
      blockage_area = False
      if blockages_present:
        count = 0
        for _, (x_range,y_range) in blockages:
            if (x_range[0] < x < x_range[1]) and (y_range[0] < y < y_range[1]):
              blockage_area = True
            count+=1
      node_h = self.grid.get_node(bottom_layer, x, y, True) 
      if blockage_area :
        node_h.add_current(0)
        J[node_h.G_loc] += 0
      else:
        node_h.add_current(-I)
        J[node_h.G_loc] += -I
      
      self._J_nodes.append(node_h)
        
    for vsrc_num,vsrc_node in enumerate(vsrc_nodes):
      V = vsrc_node[2]
      J[num_nodes+vsrc_num] = V
    
    return sparse.dok_matrix(J)

  def update_J(self, current_map, vsrc_nodes, blockages=None):
    for node_h in self._J_nodes:
      node_h.set_current(0)
    self._J = self.create_J(current_map, vsrc_nodes, blockages)

  def set_J(self,J):
    self._J = J

  def build(self, current_map, vsrc_nodes, grid = None, blockages= None):
    if grid == None:
      self.grid.build(blockages)
      self.generate_Vsrc_nodes(vsrc_nodes)
    elif grid != None:
      self.grid = grid
      print("ERROR: unknown condition")
    self._G = self.grid.Gmat 
    self._J = self.create_J(current_map,vsrc_nodes, blockages)

  def solve_ir(self,
                ir_map_file = None,
                J_map_file = None,
                grid= None,
                regions=None):
    st  = time();
    if grid is None:
      G = sparse.dok_matrix.tocsc(self._G)
    else:
      G = sparse.dok_matrix.tocsc(grid.Gmat)  
    st  = time();
    I = sparse.identity(self._G.shape[0]) * 1e-13
    G = G + I
    st  = time();
    J = sparse.dok_matrix.tocsc(self._J)
    st  = time();
    V = sparse_algebra.spsolve(G, J, permc_spec='COLAMD', use_umfpack=False)# , permc_spec=None, use_umfpack=True)
    st  = time();
    nodes = self.grid.nodes
    solution = np.zeros((0,3))
    current = np.zeros((0,3))
    unit_micron = self.grid_params['unit_micron']
    worst_case_ir = 0
    if regions is not None:
        region_ir = np.zeros(len(regions))
    else:
        region_ir = None
    if ir_map_file != None or  J_map_file != None:
      for node_num,node_h in enumerate(nodes):
        if(node_h.has_stripe): 
          node_h.set_voltage(V[node_num]) 
          if(node_h.layer == self.grid.bottom_layer):
            ir_drop =  self.vdd - V[node_num]
            solution = np.append(solution,
                               [[node_h.x/unit_micron, 
                                 node_h.y/unit_micron, 
                                 self.vdd - V[node_num]]],
                               axis=0)
            current = np.append(current,[[node_h.x/unit_micron, 
                                 node_h.y/unit_micron, 
                                 -node_h.current]],
                               axis=0)
            if regions is not None:
              for n, region in enumerate(regions):
                  x_range = region[0]
                  y_range = region[1]
                  if(    (x_range[0] < node_h.x < x_range[1]) 
                     and (y_range[0] < node_h.y < y_range[1])):
                     region_ir[n] = max(region_ir[n], ir_drop)
                     break
       
      worst_case_ir = np.max(solution[:,2])
    else:
      for node_num,node_h in enumerate(nodes):
        if(node_h.has_stripe and node_h.layer == self.grid.bottom_layer):
          ir_drop =  self.vdd - V[node_num]
          worst_case_ir = max(worst_case_ir, ir_drop) 
          if regions is not None:
            for n, region in enumerate(regions):
                x_range = region[0]
                y_range = region[1]
                if(    (x_range[0] < node_h.x < x_range[1]) 
                   and (y_range[0] < node_h.y < y_range[1])):
                   region_ir[n] = max(region_ir[n], ir_drop)
                   break
          

    st  = time();
    if ir_map_file != None:
      np.savetxt(ir_map_file, solution, fmt='%8.3f, %8.3f, %8.6e') 
    if J_map_file != None:
      np.savetxt(J_map_file, current, fmt='%8.3f, %8.3f, %8.6e') 
    return solution, worst_case_ir, region_ir
