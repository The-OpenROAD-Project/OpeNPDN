
import bisect
from scipy.sparse import dok_matrix
from pprint import pprint
import numpy as np
from T6_PSI_settings import T6_PSI_settings
import pickle

class node:
    def __init__(self):
        self.G_loc =0;
        self.x = 0  
        self.y = 0
        self.l = 0
        self.V = 0
        self.J = 0
    def set_G_loc(self,loc):
        self.G_loc = loc
    def get_G_loc(self):
        return self.G_loc;
    def set_loc(self,l,x,y):
        self.x = x
        self.y = y
        self.l = l
    def get_layer(self):
        return self.l;
    def get_loc(self):    
        return (self.l,self.x,self.y)
    def set_voltage(self,V):
        self.V = V
    def get_voltage(self):
        return self.V
    def set_current(self,J):
        self.J = J
    def add_current(self,J):
        self.J = self.J+J
    def get_current(self):
        return self.J
    def print(self):
        print("Node num %d"%self.G_loc)
        print("\t Location: x %d y %d layer %d "%(self.x,self.y,self.l))
        print("\t Voltage: %f Current: %f "%(self.V,self.J))

class template:
    def __init__(self,num_layers):
        self.num_nodes = 0;
        self.num_layers = num_layers
        self.node_map = {}
        self.nodes = []
        self.node_map_x = {}
        self.node_map_y = {}
        for n in range(0,num_layers):
            self.node_map[n] = {}
            self.node_map_x[n] = []
            self.node_map_y[n] = []

    def insert_node(self,layer,x,y):
        if x in self.node_map[layer]:
            if y in self.node_map[layer][x]:
                return self.node_map[layer][x][y]
        else:
            self.node_map[layer][x] = {}        
        node_h = node();
        node_h.set_loc(layer,x,y)
        node_h.set_G_loc(self.num_nodes)
        self.node_map[layer][x][y] = node_h
        self.nodes.append(node_h)
        idx = bisect.bisect_left(self.node_map_x[layer],x)
        if idx != len(self.node_map_x[layer]) and self.node_map_x[layer][idx] != x:
            self.node_map_x[layer].insert(idx,x)
        elif idx ==len(self.node_map_x[layer]) :
            self.node_map_x[layer].append(x)
        idx = bisect.bisect_left(self.node_map_y[layer],y)
        if idx != len(self.node_map_y[layer]) and self.node_map_y[layer][idx] != y:
            self.node_map_y[layer].insert(idx,y)
        elif idx ==len(self.node_map_y[layer]) :
            self.node_map_y[layer].append(y)
        self.num_nodes = self.num_nodes+1
        return node_h
    
    def get_node_from_loc(self,loc):
        return self.nodes[loc]

    def get_node(self,layer,x,y):
        if layer != 0:     
            if x in self.node_map[layer]:
                if y in self.node_map[layer][x]:
                    return self.node_map[layer][x][y]
                else :
                    print("Error node not found")
                    return None
            else:
                print("Error node not found")
                return None
        else:
            return self.get_nearest_node(layer,x,y)

    def get_nearest_node(self,layer,x,y):
        loc_x = self.get_nearest_loc(self.node_map_x[layer],x)
        loc_y = self.get_nearest_loc(self.node_map_y[layer],y)
        return self.node_map[layer][loc_x][loc_y]


    def get_nearest_loc(self, in_list,val):
        
        idx_l = bisect.bisect_left(in_list,val)
        idx_r = bisect.bisect_right(in_list,val)
        if idx_l == len(in_list):
            return in_list[idx_l-1]
        val_l = in_list[idx_l]
        if val_l == val:
            return val_l
        elif idx_l != 0:
            idx_l = idx_l-1
            val_l = in_list[idx_l]
        if idx_r == len(in_list):
            return in_list[idx_l]
        val_r = in_list[idx_r]
        #print(val_l)
        #print(val_r)
        if abs(val- val_l) < abs(val-val_r):
            loc = val_l
        else:
            loc = val_r
        return loc
    
    def print(self):
        print("template paramters:")
        print("G matrix")
        pprint(self.G_mat)
        print("Nodes:")
        for node in self.nodes:
            node.print()
        print("Nodes maps x and y:")
        pprint(self.node_map_x)
        pprint(self.node_map_y)
    def initialize_G_mat(self):
        if self.num_nodes == 0: 
            print("ERROR no objects for initialization")
        else: 
            self.G_mat = dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float64)
        
    def set_connection(self,node1,node2,cond):
        node_l1 = node1.get_G_loc()
        node_l2 = node2.get_G_loc()
        cond11 = self.get_conductance(node_l1,node_l1)
        cond12 = self.get_conductance(node_l1,node_l2)
        cond21 = self.get_conductance(node_l2,node_l1)
        cond22 = self.get_conductance(node_l2,node_l2)
        #if(node_l1 == 1724 or node_l2 == 1724):
        #    print(node_l1)
        #    print(node_l2)
        #    print("%f %f %f %f"%(cond11,cond12,cond21,cond22))
        #    pprint(cond)

        self.set_conductance(node_l1,node_l1,cond11+cond)
        self.set_conductance(node_l2,node_l2,cond22+cond)
        self.set_conductance(node_l1,node_l2,cond12-cond)
        self.set_conductance(node_l2,node_l1,cond21-cond)
        
        #if(node_l1 == 1724 or node_l2 == 1724):
        #    cond11 = self.get_conductance(node_l1,node_l1)
        #    cond12 = self.get_conductance(node_l1,node_l2)
        #    cond21 = self.get_conductance(node_l2,node_l1)
        #    cond22 = self.get_conductance(node_l2,node_l2)
        #    print(node_l1)
        #    print(node_l2)
        #    print("%f %f %f %f"%(cond11,cond12,cond21,cond22))
        #    pprint(cond)


    def get_conductance(self,row_idx,col_idx):
        #TODO condition to check if out of bounds
        return self.G_mat[row_idx,col_idx]
    def set_conductance(self,row_idx,col_idx,cond):
        #TODO condition to check if out of bounds
        self.G_mat[row_idx,col_idx] = cond
    def get_G_mat(self):
        return self.G_mat

def create_templates():
    settings_obj = T6_PSI_settings.load_obj()
        
    width_values = []
    res_per_l = []
    via_res = []
    dirs = []
    min_width = []
    pitches = []
    # Store a local copy of the variables from the settings_obj object from the
    # information in the JSON file
    for layer in settings_obj.PDN_layers_ids:
        attributes = settings_obj.LAYERS[layer]
        width_values.append(attributes['width'])
        min_width.append(attributes['min_width'])
        res_per_l.append(attributes['res'])
        dirs.append(attributes['direction'])
        pitches.append(attributes['pitch'])
    layer = settings_obj.TECH_LAYERS[0]
    via_res_1 = settings_obj.LAYERS[layer]['via_res']

    # Create the template with multiple layers based on the combination of
    # pitches of layers in the JSON
    for l in range(1, settings_obj.NUM_LAYERS ):
        layer = settings_obj.TECH_LAYERS[l]
        if layer in settings_obj.PDN_layers_ids:
            via_res.append(via_res_1)
            via_res_1 = settings_obj.LAYERS[layer]['via_res']
        else:
            via_res_1 += float(
                settings_obj.LAYERS[layer]['via_res'])

    width_values = np.round(np.array(width_values)*settings_obj.lef_unit)
    res_per_l = np.array(res_per_l)
    via_res = np.array(via_res)
    # Set dir = 1 for those layers in the PDN which are veritcal
    dirs = np.array([(d == "V") for d in dirs]) * 1
    rho_values = res_per_l * min_width * 1e6
    # Setting the pitch values for every layer in the template
    pitch_values = np.zeros(
        (settings_obj.NUM_TEMPLATES, settings_obj.NUM_PDN_LAYERS))
    template_layers = []
    for p, layer_pitch in enumerate(pitches):
        num_layer_pitches = len(layer_pitch)
        if num_layer_pitches <= 0:
            print("ERROR: pitch of a PDN layer undefined")
            sys.exit()
        elif num_layer_pitches == 1:
            pitch_values[:, p] = layer_pitch[-1]
        else:
            template_layers.append(p)
    for template_num in range(settings_obj.NUM_TEMPLATES):
        for p in template_layers:
            pitch_values[template_num, p] = pitches[p][template_num]
    template_num +=1
    pitch_values= np.round(pitch_values*settings_obj.lef_unit)
    if template_num != settings_obj.NUM_TEMPLATES :
        print(template_num)
        print(
            "ERROR: number of templates generated does not match number provided.",
            "Please check the template_definition.json file")
        sys.exit()
    template_list = []
    #TODO make offsets
    offsets = np.zeros((pitch_values.shape[0],pitch_values.shape[1]))*settings_obj.lef_unit
    # Call to the template definition which calls the function that creates G
    #TODO temp 
    LENGTH_REGION = settings_obj.LENGTH_REGION*settings_obj.lef_unit
    WIDTH_REGION = settings_obj.WIDTH_REGION*settings_obj.lef_unit
    for temp_num, pitch_template in enumerate(pitch_values):
        template_obj = template_def(settings_obj.NUM_PDN_LAYERS, pitch_template,
                                    width_values, rho_values, via_res,
                                    dirs,offsets[temp_num],
                                    LENGTH_REGION,
                                    WIDTH_REGION)
        print("Creating Template %d " % temp_num)
        template_list.append(template_obj)
        dirname = settings_obj.template_file
        fname = dirname + "/template_obj_%d.pkl" % temp_num
        with open(fname, 'wb') as template_file:
            pickle.dump(template_obj, template_file)
    return template_list

#TODO will break if top two layers are not in oposite directions
def template_def(num_layers,pitches,widths,rhos,via_res,dirs,offsets,LENGTH,WIDTH):
    template_obj = template(num_layers)
    #create the nodes
    #layer 1
    offset = np.round(offsets[0])
    pitch = np.round(pitches[0])
    x = offset
    while(x<LENGTH):
        y = offset
        while(y<WIDTH):
            template_obj.insert_node(0,x,y) 
            y=y+pitch           
        x = x+pitch

    #layers 2 to end
    for i in range(0,num_layers-1):
        for k in range(i + 1, num_layers):
            if not dirs[k] == dirs[i]:
               break
        if dirs[i] == 0:
            offset_x = np.round(offsets[k])
            offset_y = np.round(offsets[i])
            pitch_x = np.round(pitches[k])
            pitch_y = np.round(pitches[i])
        else:
            offset_x = np.round(offsets[i])
            offset_y = np.round(offsets[k])
            pitch_x = np.round(pitches[i])
            pitch_y = np.round(pitches[k])
        x = offset_x
        while(x<LENGTH):
            y = offset_y
            while(y<WIDTH):
                if i != 0:
                    template_obj.insert_node(i,x,y) 
                template_obj.insert_node(k,x,y) 
                y=y+pitch_y           
            x = x+pitch_x
            
    #initialize G matrix after all nodes have been created
    template_obj.initialize_G_mat()
    #template_obj.print()
    #create stripes
    # layers stripes
    for i in range(0,num_layers):
        rho = rhos[i]
        width = widths[i]
        #print(dirs[i])
        if dirs[i]==0:
            for y in template_obj.node_map_y[i]:
                for n_x, x in enumerate(template_obj.node_map_x[i]):
                    if n_x != 0:
                        x_prev = template_obj.node_map_x[i][n_x-1]
                        node_1 = template_obj.get_node(i,x,y) 
                        node_2 = template_obj.get_node(i,x_prev,y) 
                        cond = rho * (x-x_prev) / width
                        cond = 1/cond
                        #if(node_1.get_G_loc() == 1724):
                        #    print("hor")
                        #    print(x)
                        #    print(x_prev)
                        #    print(y)
                        #    print(node_1.get_loc())
                        template_obj.set_connection(node_1,node_2,cond)
        else:
            for x in template_obj.node_map_x[i]:
                for n_y, y in enumerate(template_obj.node_map_y[i]):
                    if n_y != 0:
                        y_prev = template_obj.node_map_y[i][n_y-1]
                        node_1 = template_obj.get_node(i,x,y) 
                        node_2 = template_obj.get_node(i,x,y_prev) 
                        cond = rho * (y-y_prev) / width 
                        cond = 1/cond
                        #if(node_1.get_G_loc() == 1724):
                        #    print("verty")
                        #    print(x)
                        #    print(y)
                        #    print(y_prev)
                        template_obj.set_connection(node_1,node_2,cond)
    #vias  
    for i in range(0,num_layers-1):
        for k in range(i + 1, num_layers):
            if not dirs[k] == dirs[i]:
               break
        res = via_res[i]
        if dirs[i] == 0:
            offset_x = offsets[k]
            offset_y = offsets[i]
            pitch_x = pitches[k]
            pitch_y = pitches[i]
        else:
            offset_x = offsets[i]
            offset_y = offsets[k]
            pitch_x = pitches[i]
            pitch_y = pitches[k]
        x = offset_x
        while(x<LENGTH):
            y = offset_y
            while(y<WIDTH):
                #via reistance
                node_1 = template_obj.get_node(i,x,y) 
                node_2 = template_obj.get_node(i+1,x,y) 
                cond_via = 1/res
                template_obj.set_connection(node_1,node_2,cond_via)
                y=y+pitch_y           
            x = x+pitch_x
    #template_obj.print()
    return template_obj

def load_templates():   
    template_list = []
    settings_obj = T6_PSI_settings.load_obj()
    for temp_num in range(settings_obj.NUM_TEMPLATES):
        print("Loading Template %d " % temp_num)
        dirname = settings_obj.template_file
        fname = dirname + "/template_obj_%d.pkl" % temp_num
        with open(fname, 'rb') as template_file:
            template_obj = pickle.load(template_file)
            template_list.append(template_obj)
    return template_list

def main():
    #template_obj = template(5)
    #node_1 = template_obj.insert_node(3,4,5)
    #node_2 = template_obj.insert_node(2,3,4)
    #template_obj.insert_node(1,2,3)
    #template_obj.initialize_G_mat()
    #template_obj.set_connection(node_1,node_2,5.6)
    #template_obj.print()
    create_templates()
    #template_obj =     template_def(num_layers,pitches,widths,via_res,rhos,dirs,offsets,LENGTH,WIDTH):
    #template_obj.print()

    
if __name__ == '__main__':
    main()

#class irsolver():
#    def __init__():
#        read_C4_data()
#        create_Gmat()
#        create_J()
#        add_C4_loc()
    
                
         
        

    

