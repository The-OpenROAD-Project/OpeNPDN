import re
import sys
import os
import csv
from scipy import ndimage
import numpy as np
from T6_PSI_settings import T6_PSI_settings
import current_map_generator as CMG


def process_voltage(cell_data,IR_file):
    settings_obj = T6_PSI_settings()
    print('Reading IR map')
    with open('./output/IR_drop.csv', 'rb') as infile:
        IR_drop = np.loadtxt(infile, delimiter=",")
        V = settings_obj.VDD - IR_drop 
    width1 = int(cell_data['area'][1][0])
    height1 = int(cell_data['area'][1][1])
    #print("area %f %f"%(cell_data['area'][1][0],cell_data['area'][1][1]))
    width2 = int(settings_obj.WIDTH_REGION*settings_obj.NUM_REGIONS_X*1e6)
    height2 = int(settings_obj.LENGTH_REGION*settings_obj.NUM_REGIONS_Y*1e6)
    width = max(width1,width2)
    height = max(height1,height2)
    #print(IR_drop.shape)
    assert width == IR_drop.shape[0], (
    "Loaded map does not match template width, template %d IR map %d"%( 
                                                   width, IR_drop.shape[0])) 
    assert height == IR_drop.shape[1], (
    "Loaded map does not match template height, template %d IR map %d"%( 
                                                   height,IR_drop.shape[1])) 
    #print(settings_obj.VDD )
    #print(V)
    V_map_0p1um = np.repeat(V.repeat(10,axis = 1),10,axis=0)
    #print("chip width %d %d"%(width,height))
    #print("chip width from def %d %d"%(width1,height1))
    for name, inst in cell_data['instances'].items():
        ll_x = int(inst['ll_x'] * 10)
        ll_y = int(inst['ll_y'] * 10)
        ur_x = int(inst['ur_x'] * 10)
        ur_y = int(inst['ur_y'] * 10)
        #print("ll x %d y %d ur x %d y %d"%(ll_x,ll_y,ur_x,ur_y))
        #area = (ur_x - ll_x) * (ur_y - ll_y)
        total_V = V_map_0p1um[ll_x:ur_x, ll_y:ur_y].sum()
        area = np.size(V_map_0p1um[ll_x:ur_x, ll_y:ur_y])
        if area == 0:
            avg_V = 0
            print("Warning cell %s not defined within the area of the IR map"%(
                        name))
        else:
            avg_V = total_V/area
        inst['V'] = avg_V
    return cell_data

def print_voltage_table(cell_data):
    with open('./output/voltage_table.txt', 'w') as outfile:
        outfile.write("Instance name, location, voltage\n")
        for name, inst in cell_data['instances'].items():
            outfile.write("%s , (%3.2f, %3.2f) , %5.4f\n"%(name, inst['ll_x'], 
                                                inst['ll_y'], inst['V']))
        
    

def main():
    settings_obj = T6_PSI_settings()

    if len(sys.argv) != 5 :
        print("ERROR Insufficient arguments")
        print(
            "Enter the full path names of the DEF, LEF and power report files")
        print(" Format resolution_mapping.py <def_file> <\"lef_file1 lef_file2\"> <power_rpt> <IR_drop_map>")
        sys.exit(-1)

    def_file = sys.argv[1]
    lef_files = sys.argv[2]
    power_file = sys.argv[3]
    IR_file = sys.argv[4]
    if not os.path.isfile(def_file):
        print("ERROR unable to find " + def_file)
        sys.exit(-1)
    lef_files = lef_files.split();
    for i in range(len(lef_files)):
        if not os.path.isfile(lef_files[i]):
            print("ERROR unable to find " + lef_files[i])
            sys.exit(-1)
    if not os.path.isfile(power_file):
        print("ERROR unable to find " + power_file)
        sys.exit(-1)
    if not os.path.isfile(IR_file):
        print("ERROR unable to find " + IR_file)
        sys.exit(-1)
    cell_data = CMG.create_cell_data(def_file, lef_files, power_file)
    cell_data_V = process_voltage(cell_data, IR_file)
    print_voltage_table(cell_data_V)

if __name__ == '__main__':
    main()

