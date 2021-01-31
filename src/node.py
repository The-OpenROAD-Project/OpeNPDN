class node:
  def __init__(self):
    self._G_loc =0;
    self._x = 0  
    self._y = 0
    self._l = 0
    self._V = 0
    self._J = 0
    self._upper = None
    self._lower = None
    self._stripe = False
    self._blockage = False
  
  @property
  def has_stripe(self):
    return self._stripe

  @property
  def has_upper(self):
    return self._upper != None

  @property
  def has_lower(self):
    return self._lower != None

  @property
  def upper_node(self):
    return self._upper

  @property
  def lower_node(self):
    return self._lower 

  @property
  def x(self):
    return self._x 
    
  @property
  def y(self):
    return self._y 
    
  @property
  def layer(self):
    return self._l 

  @property
  def current(self):
    return self._J

  @property
  def voltage(self):
    return self._V
  
  @property
  def G_loc(self):
    return self._G_loc;
  
  @property
  def location(self):    
    return (self._l,self._x,self._y)

  @property 
  def blockage(self):
    return self._blockage

  def set_blockage(self,blockage):
    self._blockage =blockage

  def set_G_loc(self,loc):
    self._G_loc = loc
  
  #def get_G_loc(self):
  #  return self._G_loc;
  
  def set_location(self,l,x,y):
    self._x = x
    self._y = y
    self._l = l
  
  #def get_layer(self):
  #  return self._l;
  
  #def get_loc(self):    
  #  return (self._l,self._x,self._y)
  
  def set_voltage(self,V):
    self._V = V
  
  #def get_voltage(self):
  #  return self._V
  
  def set_current(self,J):
    self._J = J
  
  def add_current(self,J):
    self._J = self._J+J
  
  #def get_current(self):
  #  return self._J

  def set_upper(self,upper_node):
    self._upper = upper_node 

  def set_lower(self,lower_node):
    self._lower = lower_node 

  def set_stripe(self,stripe):
    self._stripe = stripe

  def print(self):
    print("Node num %d"%self._G_loc)
    print("\t Location: x %d y %d layer %d "%(self._x,self._y,self._l))
    print("\t Voltage: %f Current: %f "%(self._V,self._J))

