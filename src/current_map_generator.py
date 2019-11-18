#BSD 3-Clause License
#
#Copyright (c) 2019, The Regents of the University of Minnesota
#
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#!/usr/bin/python

"""
Created on Wed May  8 22:10:01 2019

@author:Vidya A Chhabria
"""

import re
import sys
import os
import csv
from scipy import ndimage
import numpy as np
from T6_PSI_settings import T6_PSI_settings


def read_power_report(pwr_file):
    print('Reading power report file ')
    power_rep = {}
    unit = {'m': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12}
    with open(pwr_file) as f:
        comp_key = ""
        for line in f:
            if re.match(r'^[\t ]*Instance:', line, flags=re.IGNORECASE):
                _, comp_key = line.split(':', 1)
                comp_key = comp_key.strip()
                power_rep[comp_key] = {}
            elif re.match(r'^[\t ]*Cell:', line, flags=re.IGNORECASE):
                _, cell = line.split(':', 1)
                power_rep[comp_key]['cell'] = cell.strip()
            elif re.match(r'^[\t ]*Liberty file:', line, flags=re.IGNORECASE):
                _, lib = line.split(':', 1)
                power_rep[comp_key]['lib'] = lib.strip()
            elif re.match(r'^[\t ]*Internal power:{0,1}[\t ]+\d',
                          line,
                          flags=re.IGNORECASE):
                data = re.findall(r'([+-]{0,1}\s+\d+\.{0,1}\d*)([mnu]{0,1})W',
                                  line)
                power_rep[comp_key]['internal_power'] = float(
                    data[0][0].strip()) * unit[data[0][1]]
            elif re.match(r'^[\t ]*Switching power:{0,1}[\t ]+\d',
                          line,
                          flags=re.IGNORECASE):
                data = re.findall(r'([+-]{0,1}\s+\d+\.{0,1}\d*)([mnu]{0,1})W',
                                  line)
                power_rep[comp_key]['switching_power'] = float(
                    data[0][0].strip()) * unit[data[0][1]]
            elif re.match(r'^[\t ]*Leakage power:{0,1}[\t ]+\d',
                          line,
                          flags=re.IGNORECASE):
                data = re.findall(r'([+-]{0,1}\s*\d+\.{0,1}\d*)([mnu]{0,1})W',
                                  line)
                power_rep[comp_key]['leakage_power'] = float(
                    data[0][0].strip()) * unit[data[0][1]]
            elif re.match(r'^[\t ]*Total power:{0,1}[\t ]+\d',
                          line,
                          flags=re.IGNORECASE):
                data = re.findall(r'([+-]{0,1}\s+\d+\.{0,1}\d*)([mnu]{0,1})W',
                                  line)
                power_rep[comp_key]['total_power'] = float(
                    data[0][0].strip()) * unit[data[0][1]]
    return power_rep



def read_def(def_file):
    print('Reading DEF file ')
    def_data = {}
    def_data['instances'] = {}
    settings_obj = T6_PSI_settings()
    with open(def_file) as f:
        components = 0
        count = 0
        comp_syn = 0
        cur_key = ""
        for line in f:

            if re.match(r'^[\t ]*DESIGN\s+[\w]+\s+;', line, flags=re.IGNORECASE):
                data = re.findall(r'[\t ]*DESIGN\s+([\w]+)\s+;', line)
                def_data['design'] = data[0]
            if re.match(r'^[\t ]*UNITS', line, flags=re.IGNORECASE):
                data = re.findall(r'[\d]+', line)
                def_data['units_per_micron'] = int(data[0])
            if re.match(r'^[\t ]*DIE', line, flags=re.IGNORECASE):
                data = re.findall(r'\([ \t]*(\d+)[ \t]*(\d+)[ \t]*\)', line)
                def_data['area'] = [
                    [
                        int(data[0][0]) / def_data['units_per_micron'],
                        int(data[0][1]) / def_data['units_per_micron']
                    ],
                    [
                        int(data[1][0]) / def_data['units_per_micron'],
                        int(data[1][1]) / def_data['units_per_micron']
                    ]
                ]
                size_x = abs(def_data['area'][0][0] - def_data['area'][1][0]) 
                size_y = abs(def_data['area'][0][1] - def_data['area'][1][1]) 
                chip_width = settings_obj.WIDTH_REGION*settings_obj.NUM_REGIONS_X*1e6
                chip_length = settings_obj.LENGTH_REGION*settings_obj.NUM_REGIONS_Y*1e6
                if (abs(size_x - chip_width) > 100 or
                    abs(size_y - chip_length) > 100 ):
                    print("ERROR: Area obtained from the DEF does not match the \
                    template definition json file. Ensure the region sizes and \
                    number of regions are defined appropriately")
                    exit(-1)
                    
            count += 1
            if re.match(r'^[\t ]*COMPONENTS \d+', line, flags=re.IGNORECASE):
                components = 1
            if re.match(r'^[\t ]*END COMPONENTS', line, flags=re.IGNORECASE):
                components = 0

            if components == 1:

                if re.match(r'^\s*-\s+[\w/]+\s+\w+', line):
                    data = re.findall(r'[\w/]+', line)
                    cur_key = data[0].strip()
                    cell = data[1].strip()
                    comp_syn = 1
                    def_data['instances'][cur_key] = {}
                    def_data['instances'][cur_key]['cell'] = cell
                if re.match(
                        r'^[\t ]*;', line
                ) and comp_syn == 1:    #semicolon at the begining of the line
                    comp_syn = 0
                    cur_key = ""

                if re.search(r'PLACED|FIXED|COVER', line,
                             flags=re.IGNORECASE) and comp_syn == 1:
                    for m in re.finditer(r'PLACED|FIXED|COVER', line):
                        loc = m.start()
                        data = re.findall(r'[\w]+', line[loc:])
                        def_data['instances'][cur_key]['ll_x'] = int(
                            data[1]) / def_data['units_per_micron']
                        def_data['instances'][cur_key]['ll_y'] = int(
                            data[2]) / def_data['units_per_micron']
                        def_data['instances'][cur_key]['orient'] = data[3]

                if (re.search(';', line)
                        and comp_syn == 1):    #semicolon at the end of the line
                    comp_syn = 0
                    cur_key = ""

    return def_data


## Parse both the power report and the DEF data, match the names
def create_cell_data(def_file, lef_files, power_file):
    print("INFO: Parsing DEF and Power report")
    def_data = read_def(def_file)
    lef_data = read_lef(lef_files)
    power_rep = read_power_report(power_file)
    number_of_cells_wo_power = 0
    number_of_cells_wo_lef = 0
    total_num = 0
    for name, inst in def_data['instances'].items():
        total_num = total_num+1
        if name in power_rep:
            p_ref = power_rep[name]
            inst['internal_power'] = p_ref['internal_power']
            inst['switching_power'] = p_ref['switching_power']
            inst['leakage_power'] = p_ref['leakage_power']
            inst['total_power'] = p_ref['total_power']
        else:
            print(
                'Warning: Cell %s has no power report in the file, defaulting to 0'
                % name)
            number_of_cells_wo_power = number_of_cells_wo_power + 1
            inst['internal_power'] = 0
            inst['switching_power'] = 0
            inst['leakage_power'] = 0
            inst['total_power'] = 0
        if inst['cell'] in lef_data:
            if (inst['orient'] in ['N', 'S', 'FN', 'FS']):
                inst['ur_x'] = inst['ll_x'] + lef_data[inst['cell']]['width']
                inst['ur_y'] = inst['ll_y'] + lef_data[inst['cell']]['height']
            elif (inst['orient'] in ['E', 'W', 'FE', 'FW']):
                inst['ur_x'] = inst['ll_x'] + lef_data[inst['cell']]['height']
                inst['ur_y'] = inst['ll_y'] + lef_data[inst['cell']]['width']
            else:
                print(
                    "Warning: Cell %s has an unexpected orientation please check the def, defaulting to 1 by 1 um"
                    % name)
                number_of_cells_wo_lef = number_of_cells_wo_lef + 1
                inst['ur_x'] = inst['ll_x'] + 1
                inst['ur_y'] = inst['ll_y'] + 1
        else:
            print(
                'Warning: Cell %s has no corresponding LEF data, defaulting to 1 by 1 um'
                % name)
            number_of_cells_wo_lef = number_of_cells_wo_lef + 1
            inst['ur_x'] = inst['ll_x'] + 1
            inst['ur_y'] = inst['ll_y'] + 1
    #TODO check if total num machtes number of components
    print("Number of cells without power = %d" % number_of_cells_wo_power)
    print("Number of cells without lef   = %d" % number_of_cells_wo_lef)
    print("Total Number of cells         = %d" % total_num) 
    return def_data


def read_lef(lef_files):
    lef_data = {}
    for i in range(len(lef_files)):
        with open(lef_files[i]) as f:
            cur_key = ""
            component = 0
            for line in f:

                if re.match(r'^[\t ]*MACRO\s+[\w]+\s*', line, flags=re.IGNORECASE):
                    data = re.findall(r'^[\t ]*MACRO\s+([\w]+)\s*', line)
                    component = 1
                    cur_key = data[0]
                    lef_data[cur_key] = {}
                    lef_data[cur_key]['cell'] = data[0]
                    #print(lef_data[cur_key]['cell'])

                if re.match(r'^[\t ]*SIZE', line,
                            flags=re.IGNORECASE) and component == 1:
                    data = re.findall(r'\d+\.{0,1}\d*', line)
                    lef_data[cur_key]['width'] = float(data[0])
                    lef_data[cur_key]['height'] = float(data[1])
                    #print(lef_data[cur_key]['width'])
                    #print(lef_data[cur_key]['height'])
                    component = 0
    return lef_data


def create_power_map(cell_data):
    settings_obj = T6_PSI_settings()
    width1 = int(cell_data['area'][1][0])
    height1 = int(cell_data['area'][1][1])
    width2 = int(settings_obj.WIDTH_REGION*settings_obj.NUM_REGIONS_X*1e6)
    height2 = int(settings_obj.LENGTH_REGION*settings_obj.NUM_REGIONS_Y*1e6)
    width = max(width1,width2)
    height = max(height1,height2)
    power_map = np.zeros((width * 10, height * 10))
    for name, inst in cell_data['instances'].items():
        ll_x = int(inst['ll_x'] * 10)
        ll_y = int(inst['ll_y'] * 10)
        ur_x = int(inst['ur_x'] * 10)
        ur_y = int(inst['ur_y'] * 10)
        area = (ur_x - ll_x) * (ur_y - ll_y)
        power_map[ll_x:ur_x, ll_y:ur_y] = power_map[
            ll_x:ur_x, ll_y:ur_y] + inst['total_power'] / area
    power_map_um = power_map.reshape(width, 10, height, 10).sum((1, 3))
    return power_map_um

def create_congest_map(congest_file,def_data):
    settings_obj = T6_PSI_settings()
    print('Reading congestion report file')
    congest_rep = {}
    res = def_data['units_per_micron']
    width = int(settings_obj.WIDTH_REGION*settings_obj.NUM_REGIONS_X*1e6)
    height = int(settings_obj.LENGTH_REGION*settings_obj.NUM_REGIONS_Y*1e6)
    congest_map = np.zeros((width * 10, height * 10))
    
    with open(congest_file) as f:
        comp_key = ""
        data = list(csv.reader(f))
        seq = iter(data)
        for row0 in seq:
            row1 = next(seq)
            # TODO works only with consequtive lines in the file. 
            ll_x = min(int(row0[0]),int(row0[2]),int(row1[0]),int(row1[2]))
            ll_y = min(int(row0[1]),int(row0[3]),int(row1[1]),int(row1[3]))
            ur_x = max(int(row0[0]),int(row0[2]),int(row1[0]),int(row1[2]))
            ur_y = max(int(row0[1]),int(row0[3]),int(row1[1]),int(row1[3]))
            ll_x = int(10*ll_x / res)
            ll_y = int(10*ll_y / res)
            ur_x = int(10*ur_x / res)
            ur_y = int(10*ur_y / res)
            if( ll_x >0 and ll_y>0 and ur_x >0 and ur_y >0 ):
                area = 100
                usd = int(row0[4]) + int(row1[4])
                tot = int(row0[5]) + int(row1[5])
                cong = usd/tot
                if cong <0 :
                    cong = 0
                congest_map[ll_x:ur_x, ll_y:ur_y] = congest_map[ll_x:ur_x,ll_y:ur_y] + cong/area
    congest_map_um = congest_map.reshape(width, 10, height, 10).sum((1, 3))
    return congest_map_um



def create_congest_map_old(congest_file,def_data):
    settings_obj = T6_PSI_settings()
    print('Reading congestion report file')
    congest_rep = {}
    res = def_data['units_per_micron']
    width = int(settings_obj.WIDTH_REGION*settings_obj.NUM_REGIONS_X*1e6)
    height = int(settings_obj.LENGTH_REGION*settings_obj.NUM_REGIONS_Y*1e6)
    congest_map = np.zeros((width * 10, height * 10))
    
    with open(congest_file) as f:
        comp_key = ""
        for line in f:
            if re.match(r'^[\t ]*\(\s*\d+\s*,\s*\d+\s*\)\s*\(\s*\d+\s*,\s*\d+\s*\)', line, flags=re.IGNORECASE):
                data = re.findall(r'\([ \t]*(\d+)[ \t]*,[ \t]*(\d+)[ \t]*\)', line)
                ll_x = int(10*int(data[0][0]) / def_data['units_per_micron'])
                ll_y = int(10*int(data[0][1]) / def_data['units_per_micron'])
                ur_x = int(10*int(data[1][0]) / def_data['units_per_micron'])
                ur_y = int(10*int(data[1][1]) / def_data['units_per_micron'])
                if( ll_x >0 and ll_y>0 and ur_x >0 and ur_y >0 ):
                    area = 100
                    data = re.findall(r'[ \t]*(\d+)[ \t]*\/[ \t]*(\d+)[ \t]*', line)
                    avl = int(data[0][0]) + int(data[1][0])
                    tot = int(data[0][1]) + int(data[1][1])
                    cong = (tot -avl)/tot
                    if cong <0 :
                        cong = 0
                    congest_map[ll_x:ur_x, ll_y:ur_y] = congest_map[ll_x:ur_x,ll_y:ur_y] + cong/area
            
    congest_map_um = congest_map.reshape(width, 10, height, 10).sum((1, 3))
    return congest_map_um

def main():
    settings_obj = T6_PSI_settings()

    if len(sys.argv) != 5 and len(sys.argv) != 6 :
        print("ERROR Insufficient arguments")
        print(
            "Enter the full path names of the DEF, LEF and power report files")
        print(" Format resolution_mapping.py <def_file> <\"lef_file1 lef_file2\"> <power_rpt>")
        sys.exit(-1)

    def_file = sys.argv[1]
    lef_files = sys.argv[2]
    power_file = sys.argv[3]
    congest_file = sys.argv[4]
    if (len(sys.argv) == 6 and sys.argv[5] == "no_congestion"):
        congestion_enabled =0 
    else:
        congestion_enabled =1 
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
    if congestion_enabled == 1:
        if not os.path.isfile(congest_file):
            print("ERROR unable to find " + congest_file)
            sys.exit(-1)

    cell_data = create_cell_data(def_file, lef_files, power_file)
    power_map = create_power_map(cell_data)
    power_map = power_map *100
    print("WARNING: currents are scaled internally by a factor of 100")
    if congestion_enabled == 1:
        congest_map = create_congest_map(congest_file,cell_data)
    filtered_map = ndimage.uniform_filter(power_map, size=20, mode='mirror')

    with open(settings_obj.cur_map_process_file, 'wb') as outfile:
        np.savetxt(outfile, filtered_map, delimiter=',')
    with open(settings_obj.cur_map_file, 'wb') as outfile:
        np.savetxt(outfile, power_map, delimiter=',')
    if congestion_enabled == 1:
        with open(settings_obj.cong_map_file, 'wb') as outfile:
            np.savetxt(outfile, congest_map, delimiter=',')


if __name__ == '__main__':
    main()
