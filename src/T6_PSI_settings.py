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
"""
Created on Thu Mar 21 21:16:37 2019
This file specifies the settings that are needed for OpeNPDN
@author:Vidya A Chhabria
"""

import json
import re
import copy
import os
import sys
from collections import OrderedDict
from pprint import pprint
import pickle

import importlib.util
import numpy as np
from pprint import pprint


class T6_PSI_settings():

    def __init__(self,db,ODB_LOC,checkpoint_dir,mode):
        
        self.ODB_loc = ODB_LOC
        tech = db.getTech()
        self.lef_unit = tech.getDbUnitsPerMicron() * 1e6 # divide number by this to
        layers = tech.getLayers()

        self.work_dir = "./"
        self.temp_tcl_file =  self.work_dir + "input/PDN.cfg"
        self.temp_json_file = self.work_dir + "input/tech_spec.json"
        self.conf_json_file = self.work_dir + "input/tool_config.json"
        self.template_file = self.work_dir + "templates"
        self.cur_map_file = self.work_dir + "output/current_map.csv"
        self.cong_map_file = self.work_dir + "output/congestion_map.csv"
        self.cur_map_process_file = self.work_dir + \
                                    "work/current_map_processed.csv"
        self.map_dir = self.work_dir + "input/current_maps/"
        self.parallel_run_dir = self.work_dir + "work/parallel_runs/"
        self.CNN_data_dir =  self.work_dir + "output/"
        self.checkpoint_dir = checkpoint_dir 
        #self.checkpoint_dir = checkpoint_dir +'/checkpoint_w_cong/'
        self.checkpoint_file = 'power_grid_ckpt'
        self.normalization_file = 'normalization.json'
        #self.checkpoint_dir_wo_cong = checkpoint_dir +'/checkpoint_wo_cong/'

        self.template_data = self.load_json(self.temp_json_file)
        self.config = self.load_json(self.conf_json_file)
        tcl_parser_obj = tcl_parser(self.temp_tcl_file)


        self.NUM_VDD = self.config["num_vdd_per_region"]
        self.current_unit = self.config["current_unit"]
        self.num_parallel_runs = self.config["num_parallel_runs"]
        self.num_per_run = self.config["num_per_run"]
        self.num_maps = self.config["num_maps"]
        self.start_maps = self.config["start_maps"]
        self.validation_percent = self.config["validation_percent"]
        self.test_percent = self.config["test_percent"]
        self.N_EPOCHS = self.config["N_EPOCHS"]
        self.max_current = self.config["max_current"]
        self.current_map_num_regions = self.config["current_map_num_regions"]

        stdcells_list = tcl_parser_obj.list_grid_stdcell()
        self.template_names_list = stdcells_list
        temp0 = tcl_parser_obj.get_grid_stdcell(stdcells_list[0])
        chip = db.getChip()
        if mode == 'INFERENCE':
            block = chip.getBlock()
            unit_micron = 1/block.getDefUnits()

            die_area = block.getDieArea()
            size_x = abs(die_area.xMax() - die_area.xMin())*unit_micron 
            size_y = abs(die_area.yMax() - die_area.yMin())*unit_micron 
        else:
            size_x = self.template_data['property']['SIZE_REGION_X']
            size_y = self.template_data['property']['SIZE_REGION_Y']
        size_x *= 1e3
        size_y *= 1e3
        size_x =round(size_x)*1e-9 
        size_y =round(size_y)*1e-9
        self.WIDTH_REGION = size_x
        self.LENGTH_REGION = size_y
        self.NUM_TEMPLATES = len(stdcells_list)
        self.PDN_layers_ids = temp0.list_layers()

        self.NUM_REGIONS_X = self.template_data['property']['NUM_REGIONS_X']
        self.NUM_REGIONS_Y = self.template_data['property']['NUM_REGIONS_Y']
        self.NUM_REGIONS = self.NUM_REGIONS_X * self.NUM_REGIONS_Y

        self.LAYERS = {}
        self.TECH_LAYERS =[]
        for layer in layers:
            layer_num = layer.getRoutingLevel()
            if layer_num < 1:
                continue
            layer_name = layer.getName()
            self.TECH_LAYERS.append(layer_name)
            self.LAYERS[layer_name] = {}
            self.LAYERS[layer_name]['min_width'] = layer.getWidth() / self.lef_unit
            self.LAYERS[layer_name]["width"] = 0 # default value? layer.getWidth() ?
            self.LAYERS[layer_name]['pitch'] = 0 # default value? layer.getPitch() ?
            self.LAYERS[layer_name]['t_spacing'] = layer.getPitch() / self.lef_unit

            layer_res = layer.getResistance()
            if(layer_res <=1e-2):
                self.LAYERS[layer_name]['res'] = self.template_data['layers'][str(layer_num)]['res']
            else:
                self.LAYERS[layer_name]['res'] = layer.getResistance()
            #print(layer.getResistance())
            via_layer = layer.getUpperLayer()
            via_res = self.template_data['layers'][str(layer_num)]['via_res']
            if(via_layer != None):
                if(via_layer.getResistance() >=1e-2):
                    via_res =via_layer.getResistance() 
            self.LAYERS[layer_name]['via_res'] = via_res

            if layer.getDirection() == 'VERTICAL' or layer.getDirection() == 'V':
                layer_dir = 'V'
            elif layer.getDirection() == 'HORIZONTAL' or layer.getDirection() == 'H':
                layer_dir = 'H'
            else:
                print("unknown layer direction")
                layer_dir= 'H'
            self.LAYERS[layer_name]['direction'] = layer_dir
            
        self.NUM_LAYERS = len(self.TECH_LAYERS)
        for layer_name in self.PDN_layers_ids:
            found = 0
            for layer in layers:
                if layer.getName() == layer_name :
                    found =1
                    break
            if found == 0:
                print("Warning layer %s not found in tech layers"%layer_name)
            layer_obj = temp0.get_layer(layer_name) 
            width = layer_obj.get_width() 
            width *=1e-6
            self.LAYERS[layer_name]['width'] = width
            pitches = []
            for stdcell_name in stdcells_list:
                temp = tcl_parser_obj.get_grid_stdcell(stdcell_name)
                layer_obj = temp.get_layer(layer_name) 
                pitch = layer_obj.get_pitch()
                pitch *= 1e-6
                pitches.append(pitch)
            self.LAYERS[layer_name]['pitch'] = pitches
        #pprint(self.LAYERS)        
        self.VDD = self.template_data['property']['VDD']
        self.n_c4 = self.NUM_VDD * self.NUM_REGIONS_X * self.NUM_REGIONS_Y
        self.IR_DROP_LIMIT = self.template_data['property']['IR_DROP_LIMIT']

        self.NUM_PDN_LAYERS = len(self.PDN_layers_ids)

    def load_json(self, json_file):
        with open(json_file) as f:
            json_data = json.load(f)
        return json_data
    def load_obj():
        file = open("./work/T6_PSI_settings.obj",'rb')
        object_file = pickle.load(file)
        file.close()
        return object_file
    def load_template_list(self):
        dirname = self.template_file
        fname = dirname + "/refined_template_list.txt" 
        return np.loadtxt(fname,dtype=int)

        
class tcl_parser():
    def __init__(self,file_name):
        self.grid_stdcells= {}
        self.build(file_name)

    def build(self,file_name):
        with open(file_name,'r') as file_file:
            for line in file_file:
                #print(line.strip())
                self.parse_line(line,file_file)
        stdcell_list = self.list_grid_stdcell()
        if "upperGrid" in stdcell_list:
            ug_obj = self.get_grid_stdcell("upperGrid")
            ug_layers = ug_obj.list_layers()
            stdcell_list = [l for l in stdcell_list if l != "upperGrid"]
            for stdcell_name in stdcell_list:
                stdcell_obj = self.get_grid_stdcell(stdcell_name)
                for layer_name in ug_layers:
                    stdcell_obj.add_layer(ug_obj.copy_layer(layer_name))
            self.grid_stdcells.pop("upperGrid")

    def parse_block(self, in_line, file_iterator):
        lines = in_line
        line = in_line
        open_brac = len(re.findall(r'{',line))
        open_brac = open_brac - len(re.findall(r'}',line))
        while open_brac>0 :
            line = next(file_iterator)
            open_brac = open_brac + len(re.findall(r'{',line))
            open_brac = open_brac - len(re.findall(r'}',line))
            lines = '\n'.join([lines,line.strip()])
        #lines = '\n'.join([lines,line.strip()])
        return lines


    def parse_line(self, line, file_iterator):
        #PARSING PDN commmands
        if re.match(r'\s*pdn\s+.*', line, flags=re.IGNORECASE):
            line_split = re.findall(r'\w+',line)
            if (line_split[1] == 'specify_grid'):
                if(line_split[2] == 'stdcell'):
                    self.parse_grid_stdcell(line,file_iterator)

    def parse_grid_stdcell(self, in_line, file_iterator):
        lines = self.parse_block(in_line, file_iterator)
        template_name, = re.findall(r'\s*name\s+(\w+)',lines)
        template_obj = std_cell(template_name)
        #size_line = re.findall(r'\s*size\s+{(\d+\.?\d*) (\d+\.?\d*)}',lines)
        #if(len(size_line)>0):
        #    size_x = size_line[0][0]
        #    size_y = size_line[0][1]
        #    template_obj.set_size(size_x, size_y)
        #else:
        #    template_obj.set_size(-1, -1)
        lines = lines.splitlines()
        line_iterator = iter(lines)
        for line in line_iterator:
            if re.match(r'\s*layers\s+{',line):
                lyr_lines= self.parse_block(line, line_iterator)
                lyr_lines = lyr_lines.splitlines()
                lyr_line_iterator = iter(lyr_lines)
                for lyr_line in lyr_line_iterator:
                    if (re.match(r'\w+\s+{',lyr_line) and 
                            not re.match(r'\s*layers\s+{',lyr_line)) :
                        mtl_lines = self.parse_block(lyr_line, lyr_line_iterator)
                        layer_name, = re.findall(r'\s*(\w+) {',lyr_line)
                        metal_obj = metal_layer(layer_name)
                        width, = re.findall(r'width\s+(\d+\.?\d*)', mtl_lines)
                        width = float(width)
                        metal_obj.set_width(width)
                        pitch, = re.findall(r'pitch\s+(\d+\.?\d*)', mtl_lines)
                        pitch = float(pitch) 
                        metal_obj.set_pitch(pitch)
                        offset, = re.findall(r'offset\s+(\d+\.?\d*)', mtl_lines)
                        offset = float(offset)
                        metal_obj.set_offset(offset)
                        template_obj.add_layer(metal_obj)
        self.specify_grid_stdcell(template_obj)
    
    
    def specify_grid_stdcell(self, stdcell_obj):
        self.grid_stdcells[stdcell_obj.name] = stdcell_obj
    
    def get_grid_stdcell(self,stdcell_name):
        return self.grid_stdcells[stdcell_name]
    
    def list_grid_stdcell(self):
        return list(self.grid_stdcells.keys())
    
    def print_stdcells(self):
        stdcells = self.list_grid_stdcell()
        print("#############################################################")
        for stdcell_name in stdcells:
            stdcell_obj = self.get_grid_stdcell(stdcell_name)
            stdcell_obj.print()
        
class std_cell():
    
    def __init__(self, name):
        self.name = name
        self.layers = OrderedDict()
        self.num_layers = 0
    
    #def set_size(self,size_x, size_y):
    #    self.size = (float(size_x), float(size_y))
    #
    #def get_size(self):
    #    return self.size
    
    def add_layer(self, layer):
        layer.set_postion(self.num_layers)
        self.layers[layer.name] = layer
        self.num_layers = self.num_layers + 1
    
    def get_layer(self, layer_name):
        return self.layers[layer_name]
    
    def copy_layer(self, layer_name):
        return copy.deepcopy(self.layers[layer_name])
    
    def list_layers(self):
        return list(self.layers.keys())
    
    def print(self):
        print("STD Cell         : %s "%self.name)
        print("#############################################################")
        #print("Size             : (%5.1f, %5.1f) "%self.get_size())
        print("Number of Layers : %d"%self.num_layers)
        print("Layers           : (%s)"%(",".join(self.list_layers())))
        print("Layer Descriptions:")
        print("*************************************************************")
        layers = self.list_layers()
        for layer_name in layers:
            layer_obj = self.get_layer(layer_name)
            layer_obj.print()
        print("*************************************************************")
        print("#############################################################")
        
class metal_layer():
    def __init__(self, name):
        self.name = name 
        self.position = -1
    
    def set_width(self, width):
        self.width = float(width)
    
    def get_width(self):
        return self.width
    
    def set_pitch(self, pitch):
        self.pitch = float(pitch)
    
    def get_pitch(self):
        return self.pitch
    
    def set_offset(self, offset):
        self.offset = float(offset)
    
    def get_offset(self):
        return self.offset
    
    def set_postion(self, position):
        self.position = position
    
    def get_position(self):
        return self.position
    
    def print(self):
        print("Layer Name       : %s "%self.name)
        print("===================================")
        print("Pitch            : %5.1f"%self.get_pitch())
        print("Width            : %5.1f"%self.get_width())
        print("Offset           : %5.1f"%self.get_offset())
        print("Position         : %5.1f"%self.get_position())
        print("===================================")

        
if __name__ == '__main__':
    if(len(sys.argv) != 4 and len(sys.argv) != 5):
        print("ERROR: Settings requires either 4 or 5 input arguments")
        sys.exit(-1)
    odb_loc = sys.argv[1]  
    checkpoint_dir = sys.argv[2]  
    mode = sys.argv[3]  
    if mode == 'TRAIN':
        if len(sys.argv) != 5:
            print("ERROR: Training mode requires atleast 4 input arguments")
        print("OpeNPDN Training Mode:")
        lef_list = sys.argv[4]  
        lef_files = lef_list.split();
        for i in range(len(lef_files)):
            if not os.path.isfile(lef_files[i]):
                print("ERROR unable to find " + lef_files[i])
                sys.exit(-1)
        spec = importlib.util.spec_from_file_location("opendbpy",odb_loc )
        odb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(odb)
        db = odb.dbDatabase.create()
        odb.odb_read_lef(db,lef_files)
    elif mode == 'INFERENCE':
        print("OpeNPDN Inference Mode:")
        spec = importlib.util.spec_from_file_location("opendbpy",odb_loc )
        odb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(odb)
        db = odb.dbDatabase.create()
        db = odb.odb_import_db(db, "./work/PDN.db")
        if db == None:
            exit("Import DB Failed, check work/PDN.db")

    else:  
        print("MODE not recognize, possible inputs are \'TRAIN\' or \'INFERENCE\'")
        exit(-1)

    obj = T6_PSI_settings(db,odb_loc,checkpoint_dir,mode)
    filehandler = open(obj.work_dir+"work/T6_PSI_settings.obj","wb")
    pickle.dump(obj,filehandler)
    filehandler.close()
    #obj.print_stdcells()
