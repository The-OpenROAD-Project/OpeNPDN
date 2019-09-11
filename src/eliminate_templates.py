import numpy as np
import math 
import os.path
from T6_PSI_settings import T6_PSI_settings
from construct_eqn import construct_eqn
from create_template import define_templates
from scipy import sparse as sparse_mat

def eliminate_templates(template_list):
    settings_obj = T6_PSI_settings()
    dirname = settings_obj.template_file
    fname = dirname + "/refined_template_list.txt" 
    refined_template_list = redundant_template_elimination(template_list)
    np.savetxt(fname,refined_template_list,fmt='%d')
    return refined_template_list

def load_template_list():
    settings_obj = T6_PSI_settings()
    dirname = settings_obj.template_file
    fname = dirname + "/refined_template_list.txt" 
    if not os.path.isfile(fname):
        print("######################################################")
        print("Refined template file not found. Run eliminate_templates.py")
        print("######################################################")
    refined_template_list = np.loadtxt(fname,dtype=int)
    return refined_template_list

def redundant_template_elimination(template_list):
    settings_obj = T6_PSI_settings()
    eq_obj = construct_eqn()
    max_drop = settings_obj.VDD*np.ones(len(template_list))
    temp_util = np.zeros(len(template_list))

    total_tracks = 0
    num_tracks_layer = {}
    for i, layer in enumerate(settings_obj.TECH_LAYERS):
        attributes = settings_obj.LAYERS[layer]
        dirs = attributes['direction']
        if dirs == 'V':
            num_tracks_layer[layer] = math.floor(
                settings_obj.WIDTH_REGION / attributes['t_spacing'])
        else:
            num_tracks_layer[layer] = math.floor(
                settings_obj.LENGTH_REGION / attributes['t_spacing'])
        total_tracks = total_tracks + num_tracks_layer[layer]


    for template_num, template_obj in enumerate(template_list):
        temp_util[template_num] = pdn_util(settings_obj,template_list,
                                                        template_num,
                                                        num_tracks_layer,
                                                        total_tracks)

    size_x = int(settings_obj.WIDTH_REGION)
    size_y = int(settings_obj.LENGTH_REGION)
    current_map = 1e-6 * np.ones((size_x+20,size_y+20))
    regional_current,_ = eq_obj.get_regional_current(
                current_map, 0, 0)

    for template_num, template_obj in enumerate(template_list):
        g_start = template_obj.start
        G = template_obj.G
        J = eq_obj.create_J(regional_current, template_obj)
        G, J = eq_obj.add_vdd_to_G_J(G, J, template_obj, 0)
        J = sparse_mat.dok_matrix(J)
        solution = eq_obj.solve_ir(G, J)
        bot = g_start[0] 
        top = g_start[1]
        V = solution[int(bot):int(top)]
        max_drop[template_num] = max(settings_obj.VDD - V)
    template_list_updated =[]
    print(temp_util)
    for num_i,util_i in enumerate(temp_util):
        drop_i = max_drop[num_i]
        eliminate = 0 ;
        for num_j,util_j in enumerate(temp_util):
            drop_j = max_drop[num_j]
            if (util_i>util_j and drop_i>drop_j and num_i != num_j ):
                eliminate=1
        if(not eliminate):
            template_list_updated.append(num_i)
    return template_list_updated

def pdn_util(settings_obj, template_list, template_num, num_tracks_layer,
             total_tracks):
    template_obj = template_list[template_num]
    util_layer = np.zeros(template_obj.pitches.shape[0])
    used_tracks = np.zeros(template_obj.pitches.shape[0])
    template_used_tracks = 0
    for i, layer in enumerate(settings_obj.PDN_layers_ids):
        attributes = settings_obj.LAYERS[layer]
        dirs = attributes['direction']
        tracks_per_stripe = math.ceil(template_obj.widths[i] /
                                      attributes['t_spacing'])
        if dirs == 'V':
            used_tracks[i] = math.floor(
                (settings_obj.WIDTH_REGION * tracks_per_stripe) /
                template_obj.pitches[i])
        else:
            used_tracks[i] = math.floor(
                (settings_obj.LENGTH_REGION * tracks_per_stripe) /
                template_obj.pitches[i])
        util_layer[i] = used_tracks[i] / num_tracks_layer[layer]
        template_used_tracks = template_used_tracks + used_tracks[i]
    template_util = (template_used_tracks / total_tracks) * 2
    return template_util

if __name__ == '__main__':
    settings_obj = T6_PSI_settings()
    template_list = define_templates(settings_obj, generate_g=0)
    eliminate_templates(template_list)
