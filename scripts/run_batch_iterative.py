#!/usr/bin/env python3

import sys
import subprocess
import numpy as np

sys.path.append('src')

from T6_PSI_settings import T6_PSI_settings

settings_obj = T6_PSI_settings();

num_of_parallel = settings_obj.num_parallel_runs
num_of_maps_per_run = settings_obj.num_per_run
start_value    = settings_obj.start_maps
num_maps  = settings_obj.num_maps
merge_file_prefix = "CNN_"

validation_percent  = settings_obj.validation_percent
test_percent        = settings_obj.test_percent      
current_unit        = settings_obj.current_unit      
VDD                 = settings_obj.VDD               
size_region_x       = settings_obj.WIDTH_REGION*1e6
size_region_y       = settings_obj.LENGTH_REGION*1e6
NUM_REGIONS_X       = settings_obj.NUM_REGIONS_X     
NUM_REGIONS_Y       = settings_obj.NUM_REGIONS_Y     
current_map_num_regions = settings_obj.current_map_num_regions

ps =[]
map_proc = 0
n=1;
while map_proc < num_maps:

    if len(ps) < num_of_parallel :
        if  map_proc+num_of_maps_per_run < num_maps :
            p=subprocess.Popen(["python3","src/generate_training_data.py","%d"%(map_proc+start_value),"%d"%(num_of_maps_per_run)])
        else:
            p=subprocess.Popen(["python3","src/generate_training_data.py","%d"%(map_proc+start_value),"%d"%(num_maps-map_proc)])
        ps.append(p)
        map_proc = map_proc + num_of_maps_per_run
        print("Launching job %d"%n)
        n =n+1;
    else:
        p = ps[0]
        p.wait()
        del ps[0]
for p in ps:
    p.wait()
print("Runs completed")

print("Reading state variables")

map_proc =0
while map_proc <num_maps:
    if  map_proc+num_of_maps_per_run < num_maps :
        state_csv_file = settings_obj.parallel_run_dir+"state_%d_to_%d.csv"%(start_value+map_proc,start_value+map_proc+num_of_maps_per_run-1)
        congest_csv_file = settings_obj.parallel_run_dir+"congest_%d_to_%d.csv"%(start_value+map_proc,start_value+map_proc+num_of_maps_per_run-1)
        current_csv_file = settings_obj.parallel_run_dir+"current_maps_%d_to_%d.csv"%(start_value+map_proc,start_value+map_proc+num_of_maps_per_run-1)
    else:
        state_csv_file = settings_obj.parallel_run_dir+"state_%d_to_%d.csv"%(start_value+map_proc,start_value+num_maps-1)
        congest_csv_file = settings_obj.parallel_run_dir+"congest_%d_to_%d.csv"%(start_value+map_proc,start_value+num_maps-1)
        current_csv_file = settings_obj.parallel_run_dir+"current_maps_%d_to_%d.csv"%(start_value+map_proc,start_value+num_maps-1)
    state = np.genfromtxt(state_csv_file, delimiter = ',')
    state= np.reshape(state,(-1,1))
    congest = np.genfromtxt(congest_csv_file, delimiter = ',')
    currents = np.genfromtxt(current_csv_file, delimiter = ',')

    if(map_proc == 0):
        state_data = state 
        congest_data = congest
        current_data = currents
    else:
        state_data = np.vstack((state_data, state))
        congest_data = np.vstack((congest_data, congest))
        current_data = np.vstack((current_data,currents))
        
    map_proc = map_proc+num_of_maps_per_run
with open( settings_obj.work_dir + 'work/'+merge_file_prefix+'state_%d_to_%d.csv' %(start_value, 
        start_value + num_maps - 1), 'wb') as outfile:
    np.savetxt(outfile,state_data, delimiter=',', fmt='%d')
with open( settings_obj.work_dir + 'work/'+merge_file_prefix+'congest_%d_to_%d.csv' %(start_value, 
        start_value + num_maps - 1), 'wb') as outfile:
    np.savetxt(outfile,congest_data, delimiter=',', fmt='%f')

print("Reading current maps")

count = 0
map_database        = np.array(current_data)
template_database   = np.array(state_data)
congest_database    = np.array(congest_data)
#map_database        = np.zeros((current_map_num_regions*current_map_num_regions*num_maps,int(3*size_region_x)*int(3*size_region_y)))
#template_database   = np.zeros((current_map_num_regions*current_map_num_regions*num_maps,1))
#congest_database    = np.zeros((current_map_num_regions*current_map_num_regions*num_maps,9))
#for i in range(start_value,start_value+num_maps):
#    power_map_file 	= settings_obj.map_dir + "current_map_%d.csv"%(i)
#    currents        = np.genfromtxt(power_map_file, delimiter = ',')
#    currents        = (currents*current_unit)/VDD
#    state = state_data[i-start_value]
#    congest = congest_data[i-start_value]
#    for n,template in enumerate(state):
#        y = int(n/ NUM_REGIONS_X)
#        x = n % NUM_REGIONS_X 
#        xcor = int(x * size_region_x)
#        ycor = int(y * size_region_y)
#        end_xcor = int(xcor + size_region_x)
#        end_ycor = int(ycor + size_region_y)
#        current_dis = currents[xcor:end_xcor, ycor:end_ycor]
#        map_database[count]         = current_dis.reshape(-1)
#        template_database[count]    = template
#        congest_database[count]     = congest[n]
#        count +=1
#
print("Creating training and validation datasets")

data_size = template_database.shape[0]
val_num   = int(validation_percent*data_size/100) 
test_num  = int(test_percent      *data_size/100)
train_num = int(data_size - val_num - test_num)

np.random.choice(data_size, size=(val_num+test_num), replace=False)

choice = np.random.choice(range(data_size), size=(val_num,), replace=False)    
ind = np.zeros(data_size, dtype=bool)
ind[choice] = True
rest = ~ind

val_currents = map_database[ind,:]
val_template = template_database[ind,:]
val_congest = congest_database[ind,:]

map_database      = map_database[rest,:]
template_database = template_database[rest,:]
congest_database  = congest_database[rest,:]
data_size         = template_database.shape[0]

choice = np.random.choice(range(data_size), size=(test_num,), replace=False)    
ind = np.zeros(data_size, dtype=bool)
ind[choice] = True
rest = ~ind

test_currents = map_database[ind,:]
test_template = template_database[ind,:]
test_congest  = congest_database[ind,:]

train_currents = map_database[rest,:]
train_template = template_database[rest,:]
train_congest  = congest_database[rest,:]

with open(settings_obj.CNN_data_dir+merge_file_prefix+"val_currents.csv"  , 'wb') as outfile:
    np.savetxt(outfile,val_currents,delimiter=',',fmt='%4.3e')
with open(settings_obj.CNN_data_dir+merge_file_prefix+"val_template.csv"  , 'wb') as outfile:
    np.savetxt(outfile,val_template,delimiter=',',fmt='%d')
with open(settings_obj.CNN_data_dir+merge_file_prefix+"val_congest.csv"  , 'wb') as outfile:
    np.savetxt(outfile,val_congest,delimiter=',',fmt='%f')
with open(settings_obj.CNN_data_dir+merge_file_prefix+"test_currents.csv" , 'wb') as outfile:
    np.savetxt(outfile,test_currents,delimiter=',',fmt='%4.3e')
with open(settings_obj.CNN_data_dir+merge_file_prefix+"test_template.csv" , 'wb') as outfile:
    np.savetxt(outfile,test_template,delimiter=',',fmt='%d')
with open(settings_obj.CNN_data_dir+merge_file_prefix+"test_congest.csv"  , 'wb') as outfile:
    np.savetxt(outfile,test_congest,delimiter=',',fmt='%f')
with open(settings_obj.CNN_data_dir+merge_file_prefix+"train_currents.csv", 'wb') as outfile:
    np.savetxt(outfile,train_currents,delimiter=',',fmt='%4.3e')
with open(settings_obj.CNN_data_dir+merge_file_prefix+"train_template.csv", 'wb') as outfile:
    np.savetxt(outfile,train_template,delimiter=',',fmt='%d')
with open(settings_obj.CNN_data_dir+merge_file_prefix+"train_congest.csv"  , 'wb') as outfile:
    np.savetxt(outfile,train_congest,delimiter=',',fmt='%f')



    
