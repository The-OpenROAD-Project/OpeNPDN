from sortedcontainers import SortedDict
from enum import IntEnum
from scipy.sparse import dok_matrix
import numpy as np
from pprint import pprint
from node import node


class Direction(IntEnum):
  HORIZONTAL = 0
  VERTICAL = 1

class grid:
  def __init__(self, grid_params):
    self._pitches = grid_params['pitches']
    self._num_layers = len(self._pitches)
    self._widths = grid_params['widths']

    self.unit_micron = grid_params['unit_micron']    
    res_per_l = grid_params['res_per_l']
    min_widths = grid_params['min_widths']
    self._rhos =  [res*w/self.unit_micron for res,w in zip(res_per_l,min_widths)]

    self._via_res = grid_params['via_res']
    self._offsets = grid_params['offsets']
    self._size_x = grid_params['size_x']
    self._size_y = grid_params['size_y']
    self._num_nodes = 0;
    self._node_map = {}
    self._nodes = []
    self._dirs = {}
    self._top_layer = self._num_layers - 1
    self._bot_layer = 0
    for n in range(0,self._num_layers):
      self._node_map[n] = SortedDict()
      self._dirs[n] = Direction(grid_params['dirs'][n])
  
  @property
  def nodes(self):
    return self._nodes

  @property
  def Gmat(self):
    return self._G_mat

  @property
  def num_nodes(self):
    return self._num_nodes
  
  @property
  def top_layer(self):
    return self._top_layer

  @property
  def bottom_layer(self):
    return self._bot_layer

  def coordinates(self,layer,x,y):
    if self._dirs[layer] == Direction.HORIZONTAL:
      loc1 = y
      loc2 = x
    else:
      loc1 = x
      loc2 = y
    return loc1,loc2 

  def insert_node(self,layer,x,y):
    loc1,loc2 = self.coordinates(layer,x,y)
    if loc1 in self._node_map[layer]:
      if loc2 in self._node_map[layer][loc1]:
        return self._node_map[layer][loc1][loc2]
    else:
      self._node_map[layer][loc1] = SortedDict()    

    node_h = node();
    node_h.set_location(layer,x,y)
    node_h.set_G_loc(self._num_nodes)
    self._node_map[layer][loc1][loc2] = node_h
    self._nodes.append(node_h)

    self._num_nodes = self._num_nodes+1
    return node_h
  
  def get_node_from_loc(self,loc):
    return self._nodes[loc]

  def get_node(self,layer,x,y,nearest=False):
    loc1,loc2 = self.coordinates(layer,x,y)
    if nearest == False:   
      if loc1 in self._node_map[layer]:
        if loc2 in self._node_map[layer][loc1]:
          return self._node_map[layer][loc1][loc2]
        else :
          print("Error node not found")
          return None
      else:
        print("Error node not found")
        return None
    else:
      return self.get_nearest_node(layer,loc1,loc2)

  def get_nearest_node(self,layer,loc1,loc2):
    keys = self._node_map[layer].keys()
    lindex = self._node_map[layer].bisect_left(loc1)
    if(lindex == 0):
      v1 = keys[lindex]
      v2 = self.get_nearest_key(self._node_map[layer][v1],loc2)
    elif lindex == len(keys):
      v1 = keys[lindex - 1]
      v2 = self.get_nearest_key(self._node_map[layer][v1],loc2)
    else:
      v1_1 = keys[lindex]
      v1_2 = keys[lindex - 1]
      v2_1 = self.get_nearest_key(self._node_map[layer][v1_1],loc2)
      v2_2 = self.get_nearest_key(self._node_map[layer][v1_2],loc2)
      if( abs(v1_1-loc1) + abs(v2_1-loc2) < abs(v1_2-loc1) + abs(v2_2-loc2) ):
        v1 = v1_1
        v2 = v2_1
      else:
        v1 = v1_2
        v2 = v2_2
    return self._node_map[layer][v1][v2]

  def get_nearest_key(self, node_map,loc):
    keys = node_map.keys()
    lindex = node_map.bisect_left(loc)
    if(lindex == 0):
      return keys[lindex]
    elif lindex == len(keys):
      return keys[lindex-1]
    else:
      val1 = keys[lindex]
      val2 = keys[lindex-1]
      if abs(loc- val1) < abs(loc-val2):
        return val1
      else:
        return val2

  def print(self):
    print("template paramters:")
    print("G matrix")
    pprint(self._G_mat)
    print("Nodes:")
    for node in self._nodes:
      node.print()

 
  def set_connection(self,node1,node2,cond):
    node_l1 = node1.G_loc
    node_l2 = node2.G_loc
    cond11 = self.get_conductance(node_l1,node_l1)
    cond12 = self.get_conductance(node_l1,node_l2)
    cond21 = self.get_conductance(node_l2,node_l1)
    cond22 = self.get_conductance(node_l2,node_l2)
    self.set_conductance(node_l1,node_l1,cond11+cond)
    self.set_conductance(node_l2,node_l2,cond22+cond)
    self.set_conductance(node_l1,node_l2,cond12-cond)
    self.set_conductance(node_l2,node_l1,cond21-cond)
  
                            


  def get_conductance(self,row_idx,col_idx):
    #TODO condition to check if out of bounds
    return self._G_mat[row_idx,col_idx]

  def set_conductance(self,row_idx,col_idx,cond):
    #TODO condition to check if out of bounds
    self._G_mat[row_idx,col_idx] = cond
  
  def create_layer_nodes(self, layer, layer_params):
    offset_x, pitch_x, offset_y, pitch_y = layer_params
    x = offset_x
    while(x<=self._size_x):
      y = offset_y
      while(y<=self._size_y):
        self.insert_node(layer,x,y) 
        y=y+pitch_y           
      x = x+pitch_x

  def create_layer_parameters(self, layer, layer_n):
    if self._dirs[layer] == Direction.HORIZONTAL:
      offset_x = np.round(self._offsets[layer_n])
      offset_y = np.round(self._offsets[layer])
      pitch_x = np.round(self._pitches[layer_n])
      pitch_y = np.round(self._pitches[layer])
    else:
      offset_x = np.round(self._offsets[layer])
      offset_y = np.round(self._offsets[layer_n])
      pitch_x = np.round(self._pitches[layer])
      pitch_y = np.round(self._pitches[layer_n])
    layer_params = (offset_x, pitch_x, offset_y, pitch_y)
    return layer_params

  def create_nodes(self):
    #create the nodes
    #layer 1 
    layer = self._bot_layer
    offset = np.round(self._offsets[self._bot_layer])
    pitch = np.round(self._pitches[self._bot_layer])
    layer_params = (offset, pitch, offset, pitch)
    self.create_layer_nodes(layer, layer_params)

    #layers 2 to end-1
    for layer in range(self._bot_layer,self._top_layer):
      for layer_n in range(layer + 1, self._top_layer+1):
        if not self._dirs[layer_n] == self._dirs[layer]:
          break
      layer_params = self.create_layer_parameters(layer,layer_n)
      if layer != self._bot_layer: 
        self.create_layer_nodes(layer, layer_params)
      if layer_n != self._top_layer:
        self.create_layer_nodes(layer_n, layer_params)
    #layer top
    layer = self._top_layer
    layer_params = self.create_layer_parameters(layer,
                                                self._bot_layer)
    self.create_layer_nodes(layer, layer_params)
   
  def initialize_G_DOK(self):
    if self._num_nodes == 0: 
      print("ERROR no objects for initialization")
    else: 
      self._G_mat = dok_matrix((self._num_nodes, self._num_nodes), dtype=np.float64)
  
  def conductivity(self,length, rho, width):
    cond = rho * length / width
    cond = 1/cond
    return  cond


  # Convention is that only the second node (node_h) gets associated with the stripe
  def create_stripe(self, layer, location, lrange):
    rho = self._rhos[layer]
    width = self._widths[layer]
    node_stripe = self._node_map[layer][location]
    initial = True
    for key in node_stripe.irange(lrange[0],lrange[1]):
      node_h = node_stripe[key]
      if initial == True:
        initial = False
      else:
        cond = self.conductivity(key - key_prev, rho, width)
        self.set_connection(node_h,node_prev_h,cond)
        node_h.set_stripe(True)
        #node_prev_h.set_stripe(True)
      key_prev = key
      node_prev_h = node_h
  
  def update_via(self, layer, node_h, add_removeN):
    sign = (2*add_removeN -1)
    node_upper = node_h.upper_node
    if node_h.has_upper and node_upper.has_stripe:
      res = self._via_res[layer]
      cond_via = (1/res) * sign
      self.set_connection(node_h,node_upper,cond_via)
    node_lower = node_h.lower_node
    if node_h.has_lower and node_lower.has_stripe:
      res = self._via_res[layer-1]
      cond_via = (1/res) * sign
      self.set_connection(node_h,node_lower,cond_via)

  def update_stripe(self, layer, location, lrange, add_removeN):
    rho = self._rhos[layer]
    width = self._widths[layer]
    node_stripe = self._node_map[layer][location]
    initial = True
    for key in node_stripe.irange(lrange[0],lrange[1]):
      node_h = node_stripe[key]
      if initial == True:
        initial = False
        idx = node_stripe.bisect_left(key)
        if idx > 0:
          key_prev = (node_stripe.keys())[idx-1]
          node_prev_h = node_stripe[key_prev]
        else:
          key_prev = key
          node_prev_h = node_h
          continue
      if( add_removeN ^ node_h.has_stripe  and node_h.blockage == False ): #remove&hasStripe,add&noStripe
        cond = self.conductivity(key - key_prev, rho, width)
        cond = cond * (2*add_removeN -1) # converto +cond -cond
        self.set_connection(node_h,node_prev_h,cond)
        node_h.set_stripe(add_removeN)
        self.update_via(layer, node_h, add_removeN)
        #node_prev_h.set_stripe(True)

      key_prev = key
      node_prev_h = node_h

  def update_layer_stripes(self, layer, x_range, y_range, new_pitch):
    lrange, sub_lrange = self.coordinates(layer, x_range, y_range)
    layer_map = self._node_map[layer]
    
    grid_pitch = self._pitches[layer]
    target_ratio = grid_pitch/new_pitch
    template_ratio = num_stripes = 0
    total_stripes =1

    for location in layer_map.irange(lrange[0],lrange[1]):
      add_removeN = (template_ratio <= target_ratio)
      num_stripes += add_removeN
      self.update_stripe(layer, location, sub_lrange, add_removeN)
      template_ratio = num_stripes/total_stripes  
      total_stripes +=1

  def update_region(self, x_range, y_range, layers_info):
    for layer, pitch in layers_info:
      self.update_layer_stripes(layer, x_range, y_range, pitch)

  def create_layer_stripes(self, layer, x_range, y_range):
    lrange, sub_lrange = self.coordinates(layer, x_range, y_range)
    layer_map = self._node_map[layer]
    for location in layer_map.irange(lrange[0],lrange[1]):
      self.create_stripe(layer, location, sub_lrange)
   
  def remove_layer_stripes(self,layer,x_range,y_range):
    lrange, sub_lrange = self.coordinates(layer, x_range, y_range)
    layer_map = self._node_map[layer]
    for location in layer_map.irange(lrange[0],lrange[1]):
      add_removeN = 0
      self.update_stripe(layer, location, sub_lrange, add_removeN)
      node_stripe = self._node_map[layer][location]
      for key in node_stripe.irange(sub_lrange[0],sub_lrange[1]):
        node_h = node_stripe[key]
        node_h.set_blockage(True)

  def remove_blockages(self, blockages):
    for layers, (x_range,y_range) in blockages:
      #print(layers)
      #print((x_range,y_range))
      for layer in layers:
        self.remove_layer_stripes(layer,x_range,y_range)

  def create_stripes(self):
    for layer in range(self._bot_layer,self._top_layer+1):
      x_range = (0, self._size_x) 
      y_range = (0, self._size_y) 
      self.create_layer_stripes(layer,x_range, y_range)


  def create_layer_vias(self, layer, layer_params):
    offset_x, pitch_x, offset_y, pitch_y = layer_params
    layer_n = layer +1
    res = self._via_res[layer]
    nearest_layer = (layer == self._bot_layer)
    nearest_layer_n = (layer_n == self._top_layer)
    x = offset_x
    while(x<self._size_x):
      y = offset_y
      while(y<self._size_y):
        #via reistance
        node_1 = self.get_node(layer,x,y,nearest_layer) 
        node_2 = self.get_node(layer_n,x,y, nearest_layer_n) 

        node_1.set_upper(node_2)
        node_2.set_lower(node_1)

        cond_via = 1/res
        self.set_connection(node_1,node_2,cond_via)
        y=y+pitch_y           
      x = x+pitch_x
  
  def create_vias(self):
    #vias  
    for layer in range(self._bot_layer,self._top_layer):
      for layer_n in range(layer + 1, self._top_layer+1):
        if not self._dirs[layer_n] == self._dirs[layer]:
          break
      res = self._via_res[layer]
      layer_params = self.create_layer_parameters(layer,layer_n)
      self.create_layer_vias(layer, layer_params)


  def build(self, blockages = None):
    self.blockages = blockages
    self.create_nodes()
    self.initialize_G_DOK()
    self.create_stripes()
    self.create_vias()
    if (blockages is not None):
      self.remove_blockages(blockages)
  
  def print_grid(self):
    for layer in range(self._bot_layer,self._top_layer):
      x_range = (0, self._size_x) 
      y_range = (0, self._size_y) 
      lrange, sub_lrange = self.coordinates(layer, x_range, y_range)
      layer_map = self._node_map[layer]
      with open("layer_map_%02d.csv"%layer,"w") as f:
        for location in layer_map.irange(lrange[0],lrange[1]):
          node_stripe = self._node_map[layer][location]
          for key in node_stripe.irange(sub_lrange[0],sub_lrange[1]):
            node_h = node_stripe[key]
            if node_h.has_stripe:
              f.write("%5.3f, %5.3f\n"%(node_h.x/self.unit_micron,node_h.y/self.unit_micron))

