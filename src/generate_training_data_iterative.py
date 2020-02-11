import sys
import numpy as np
from T6_PSI_settings import T6_PSI_settings
import random
from create_template import load_templates
from create_template import template
from create_template import node
from construct_eqn import construct_eqn
from scipy import sparse as sparse_mat
from pprint import pprint
from tqdm import tqdm

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
        if sys.argv[1] == "no_congestion":
            congestion_enabled = 0
        else:
            congestion_enabled = 1 
        map_start = 1
        num_maps = settings_obj.num_maps
        #print("Warning defaulting to %d %d and with congestion" % (map_start, num_maps))
        #print(sys.argv)
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
    template_distribution = np.zeros((len(template_list)))
    for i in tqdm(range(num_maps)):
        power_map_file = settings_obj.map_dir + "current_map_%d.csv" % (
            i + map_start)
        currents = np.genfromtxt(power_map_file, delimiter=',')
        voltage = np.zeros((0,3)) # location value tuple, (x,y,v)
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
                state.append(i)
                template_distribution[i] = template_distribution[i]+1
                max_drop.append(max_drop_region)
                current_maps.append(map_row)
                
        #IR_drop = voltage
        #IR_drop[:,2] = settings_obj.VDD - IR_drop[:,2]
        #pprint(np.flip(state_map,0))
        #with open('./output/IR_drop.csv', 'wb') as outfile:
        #    np.savetxt(outfile,IR_drop,delimiter=',')
    template_distribution = 100*template_distribution/np.sum(template_distribution)
    with open("work/distribution.txt",'w') as outfile:
        print("Percentage distribution of templates:")
        outfile.write("Percentage distribution of templates:\n")
        for i,template_name in enumerate(settings_obj.template_names_list):
            print("  %10s : %4.2f "%(template_name,template_distribution[i]))
            outfile.write("  %10s : %4.2f\n"%(template_name,template_distribution[i]))

    count = 0
    map_database        = np.array(current_maps)
    template_database   = np.array(state)
    data_size = template_database.shape[0]
    val_num   = int(settings_obj.validation_percent*data_size/100) 
    test_num  = int(settings_obj.test_percent      *data_size/100)
    train_num = int(data_size - val_num - test_num)
    
    choice = np.random.choice(range(data_size), size=(val_num,), replace=False)    
    ind = np.zeros(data_size, dtype=bool)
    ind[choice] = True
    rest = ~ind
    
    val_currents = map_database[ind,:]
    val_template = template_database[ind]
    
    map_database      = map_database[rest,:]
    template_database = template_database[rest]
    data_size         = template_database.shape[0]
    
    choice = np.random.choice(range(data_size), size=(test_num,), replace=False)    
    ind = np.zeros(data_size, dtype=bool)
    ind[choice] = True
    rest = ~ind
    
    test_currents = map_database[ind,:]
    test_template = template_database[ind]
    
    train_currents = map_database[rest,:]
    train_template = template_database[rest]
    
    with open(settings_obj.CNN_data_dir+"val_currents.csv"  , 'wb') as outfile:
        np.savetxt(outfile,val_currents,delimiter=',',fmt='%4.3e')
    with open(settings_obj.CNN_data_dir+"val_template.csv"  , 'wb') as outfile:
        np.savetxt(outfile,val_template,delimiter=',',fmt='%d')
    with open(settings_obj.CNN_data_dir+"test_currents.csv" , 'wb') as outfile:
        np.savetxt(outfile,test_currents,delimiter=',',fmt='%4.3e')
    with open(settings_obj.CNN_data_dir+"test_template.csv" , 'wb') as outfile:
        np.savetxt(outfile,test_template,delimiter=',',fmt='%d')
    with open(settings_obj.CNN_data_dir+"train_currents.csv", 'wb') as outfile:
        np.savetxt(outfile,train_currents,delimiter=',',fmt='%4.3e')
    with open(settings_obj.CNN_data_dir+"train_template.csv", 'wb') as outfile:
        np.savetxt(outfile,train_template,delimiter=',',fmt='%d')

    #with open(
    #        settings_obj.parallel_run_dir + 'max_drop_%d_to_%d.csv' %
    #    (map_start, map_start + num_maps - 1), 'w') as outfile:
    #    np.savetxt(outfile, max_drop, delimiter=',', fmt='%f')
    #with open(
    #        settings_obj.parallel_run_dir + 'state_%d_to_%d.csv' %
    #    (map_start, map_start + num_maps - 1), 'w') as outfile:
    #    np.savetxt(outfile, state, delimiter=',', fmt='%d')
    #with open(
    #        settings_obj.parallel_run_dir + 'current_maps_%d_to_%d.csv' %
    #    (map_start, map_start + num_maps - 1), 'w') as outfile:
    #    np.savetxt(outfile,current_maps, delimiter=',', fmt='%f')

if __name__ == '__main__':
    main()
