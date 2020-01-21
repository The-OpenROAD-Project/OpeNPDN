import sys
import numpy as np
from T6_PSI_settings import T6_PSI_settings
import random
from create_template_new import load_templates
from create_template_new import template
from create_template_new import node
from construct_eqn_new import construct_eqn
from scipy import sparse as sparse_mat
from pprint import pprint

def main():
    settings_obj = T6_PSI_settings.load_obj()
    if len(sys.argv) == 3:
        map_start = int(sys.argv[1])
        num_maps = int(sys.argv[2])
        congestion_enabled = 1
    elif len(sys.argv) == 4:
        map_start = int(sys.argv[1])
        num_maps = int(sys.argv[2])
        if sys.argv[3] == "no_congestion":
            congestion_enabled = 0
        else:
            congestion_enabled = 1 
    else:
        map_start = 1
        num_maps = 15
        congestion_enabled = 1 
        print("Warning defaulting to %d %d and with congestion" % (map_start, num_maps))
        print(sys.argv)
    state = [] #np.zeros((num_maps, settings_obj.NUM_REGIONS))
    max_drop = [] #np.zeros((num_maps, settings_obj.NUM_REGIONS))
    size_region_x = int(settings_obj.WIDTH_REGION * 1e6)
    size_region_y = int(settings_obj.LENGTH_REGION * 1e6)
    current_maps = []
    template_list = load_templates()
    #for i,template_obj in enumerate(template_list): 
    #    print(i)
    #    print(template_obj.G_mat.shape)
    eq_obj = construct_eqn()
    for i in range(num_maps):
        power_map_file = settings_obj.map_dir + "current_map_%d.csv" % (
            i + map_start)
        currents = np.genfromtxt(power_map_file, delimiter=',')
        currents = (currents * settings_obj.current_unit) / settings_obj.VDD
        voltage = np.zeros((0,3)) # location value tuple, (x,y,v)
        state_map = []
        max_drop_map = []
        for y in range(settings_obj.current_map_num_regions):
            for x in range(settings_obj.current_map_num_regions):
                #print("%d %d "%(x,y))
                regional_current, map_row = eq_obj.get_regional_current(currents, x, y)
                max_drop_region = 0 
                for i,template_obj in enumerate(template_list): 
                    J = eq_obj.create_J(template_obj,regional_current)
                    G_orig = template_obj.get_G_mat()
                    G, J = eq_obj.add_vdd_to_G_J(J, template_obj)
                    J = sparse_mat.dok_matrix(J)
                    solution = eq_obj.solve_ir(G, J)
                    region_voltage = eq_obj.get_regional_voltage( template_obj, solution, x, y)
                    #voltage = np.append(voltage, region_voltage,axis =0)
                    max_drop_region = max(settings_obj.VDD - region_voltage[:,2].flatten())
                    #print(i)
                    #print(max_drop_region)
                    if(max_drop_region< settings_obj.IR_DROP_LIMIT):
                        break;
                state_map.append(i)
                max_drop_map.append(max_drop_region)
                
        #IR_drop = voltage
        #IR_drop[:,2] = settings_obj.VDD - IR_drop[:,2]
        #pprint(np.flip(state_map,0))
        state.append(state_map)
        max_drop.append(max_drop_map)
        current_maps.append(currents.reshape(-1))
        #with open('./output/IR_drop.csv', 'wb') as outfile:
        #    np.savetxt(outfile,IR_drop,delimiter=',')
    with open(
            settings_obj.parallel_run_dir + 'max_drop_%d_to_%d.csv' %
        (map_start, map_start + num_maps - 1), 'w') as outfile:
        np.savetxt(outfile, max_drop, delimiter=',', fmt='%f')
    with open(
            settings_obj.parallel_run_dir + 'state_%d_to_%d.csv' %
        (map_start, map_start + num_maps - 1), 'w') as outfile:
        np.savetxt(outfile, state, delimiter=',', fmt='%d')
    with open(
            settings_obj.parallel_run_dir + 'current_maps_%d_to_%d.csv' %
        (map_start, map_start + num_maps - 1), 'w') as outfile:
        np.savetxt(outfile,current_maps, delimiter=',', fmt='%f')





if __name__ == '__main__':
    main()
