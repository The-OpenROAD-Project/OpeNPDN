from ir_solver import ir_solver
import numpy as np
from time import time
from tqdm import tqdm, trange
from os.path import dirname
import sys
sys.path.append(dirname(__file__))
import copy
import math
from matplotlib import pyplot as plt

class simulated_annealer:
  def __init__(self, initial_state, T_init, T_final, alpha_temp, \
                num_moves_per_step, signal_cong, ir_solver_h,\
                templates, regions):
    self.state = initial_state
    self.T_init = T_init
    self.T_final = T_final
    self.alpha_temp = alpha_temp
    self.num_moves_per_step = num_moves_per_step
    self.ir_solver_h = ir_solver_h
    self.signal_cong = signal_cong
    self.templates = templates
    self.regions = regions
    self.IR_DROP_LIMIT =  self.ir_solver_h.ir_drop_limit
    self.IR_PENALTY = 32/(self.IR_DROP_LIMIT/3)
    self.CONGESTION_PENALTY =16/0.1

  def sim_anneal(self, plot):
    cur_state = self.state;
    energy_hist = []
    tt=time()
    cur_energy, cur_max_drop, new_grid = self.energy(cur_state)
    energy_hist.append(cur_energy)
    self.ir_solver_h.set_grid(new_grid)
    sa_start = time();
    total = 0
    # replace with direct calculation
    T = self.T_init
    while(T>self.T_final):
      T = self.cool_down(T)
      total += 1

    T = self.T_init
    #while(T>self.T_final):
    with trange(total, position=1) as pbar:
      for _ in pbar:
        #start = time()
        for i in range(self.num_moves_per_step):               
          next_state = self.move(cur_state, self.templates)
          #tt = time()
          next_energy, next_max_drop, new_grid = self.delta_energy(
            cur_state, next_state, cur_energy, cur_max_drop)
          #tt = time()
          delta_cost =  next_energy - cur_energy
          accepted = self.acceptMove (delta_cost, T)
          if (accepted):
            self.state = next_state
            cur_energy = next_energy
            cur_state  = next_state
            self.ir_solver_h.set_grid(new_grid)
            cur_max_drop = next_max_drop
          energy_hist.append(cur_energy)
        #end = time()
        pbar.set_description("@%6.5fC %6.4fV"%(T,max(cur_max_drop)))
        #print("\r                \r@%6.5fC %6.4fV"%(T,max(cur_max_drop)),end='')
        T = self.cool_down(T)
    cur_energy, cur_max_drop, new_grid = self.energy(cur_state)
    if plot:
      plt.figure()
      plt.plot(energy_hist)

    return self.state, cur_energy  

  def pdn_util(self, template_num):
    
    t_spacing = [4800, 4800,\
                 8000, 8000,\
                 8000]
    grid_params = self.ir_solver_h.grid_params
    num_stripes = [grid_params['size_x']/4800, grid_params['size_y']/4800,\
                   grid_params['size_x']/8000, grid_params['size_y']/8000,\
                   grid_params['size_x']/8000]

    template = self.templates[template_num]
    
    util=0
    for layer, pitch in template:
      util += num_stripes[layer] * t_spacing[layer]/pitch

    total_util = util/sum(num_stripes)   

    return total_util

  def cool_down(self, T):
    T = self.alpha_temp * T
    return T

    

  def energy(self, state):
    """Calculates the length of the route."""
   
    signal_cong = self.signal_cong
    

    tot_cong = np.zeros(len(state))
    e = 0;
    template_util = np.zeros(len(state))
    for i,template_num in enumerate(state):
      template_util[i] = self.pdn_util(template_num)
      tot_cong[i] = signal_cong[i] + template_util[i]
      if(tot_cong[i]>1):
        e += self.CONGESTION_PENALTY * (tot_cong[i]-1)
      e += tot_cong[i] 

    new_grid = self.ir_solver_h.internal_grid
    for n,template in enumerate(state):
      x_range = self.regions[n][0]
      y_range = self.regions[n][1]
      layers_info = self.templates[template]
      new_grid.update_region( x_range, y_range,layers_info)

    _,max_drop, region_ir = self.ir_solver_h.solve_ir(grid = new_grid,
                                    regions=self.regions)
    max_drop = min(self.ir_solver_h.vdd,abs(max_drop)) 
    region_ir = np.minimum(np.abs(region_ir),self.ir_solver_h.vdd)

    for ir in region_ir:
      if(ir > self.IR_DROP_LIMIT):
        e += self.IR_PENALTY*(ir - self.IR_DROP_LIMIT)
  
    return e, region_ir, new_grid   
  
  def delta_energy(self, cur_state, next_state, cur_energy, cur_region_ir):
    """Calculates the length of the route."""
   
    signal_cong = self.signal_cong

    e = cur_energy;
    
    for i,template_num in enumerate(cur_state):
      if template_num != next_state[i]:
        template_util_old = self.pdn_util(template_num)
        template_util_new = self.pdn_util(next_state[i])
        tot_cong_old = signal_cong[i] + template_util_old
        tot_cong_new = signal_cong[i] + template_util_new
        if(tot_cong_old>1):
          e -= self.CONGESTION_PENALTY * (tot_cong_old - 1)
        e -= tot_cong_old
        if(tot_cong_new>1):
          e += self.CONGESTION_PENALTY * (tot_cong_new - 1)
        e += tot_cong_new
    new_grid = self.ir_solver_h.internal_grid
    for n,template in enumerate(next_state):
        x_range = self.regions[n][0]
        y_range = self.regions[n][1]
        layers_info = self.templates[template]
        new_grid.update_region( x_range, y_range,layers_info)
    _,max_drop, region_ir = self.ir_solver_h.solve_ir(regions=self.regions)
    cur_max_drop = max(cur_region_ir) 
    for ir in cur_region_ir:
      if(ir > self.IR_DROP_LIMIT):
        e -= self.IR_PENALTY*(ir - self.IR_DROP_LIMIT)
    max_drop = min(self.ir_solver_h.vdd,abs(max_drop)) 
    region_ir = np.minimum(np.abs(region_ir),self.ir_solver_h.vdd)
    for ir in region_ir:
      if(ir > self.IR_DROP_LIMIT):
        e += self.IR_PENALTY*(ir - self.IR_DROP_LIMIT)
    return e, region_ir, new_grid   

  def acceptMove(self,delta_cost, T):
    k = 1.38064852 * (10**(0))
    if(delta_cost < 0):
      return 1
    else:
      boltz_prob = math.exp((-1*delta_cost)/(k*T))
      r = np.random.uniform(0,1)
      if(r<boltz_prob):
          return 1
      else:
          return 0
                
  def move(self,state,all_templates):
    flag = 0
    """Randomly pick a template from the list of all tempaltes and insert in the current state."""
    while(flag == 0):        
      a = np.random.randint(0, len(all_templates))
      b = np.random.randint(0, len(state))
      if(state[b]!=a):
          flag = 1
    state_new = list(state)
    state_new[b] = a
    return state_new
