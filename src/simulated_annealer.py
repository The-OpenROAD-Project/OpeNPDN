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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:16:37 2019

@author: Vidya A Chhabria
"""
import math
import random
from tqdm import tqdm
import numpy as np
from scipy import sparse as sparse_mat
from T6_PSI_settings import T6_PSI_settings
from construct_eqn import construct_eqn

class simulated_annealer():

    def __init__(self, initial_state, T_init, T_final, alpha_temp,
                 num_moves_per_step, current_map,congestion_enabled):
        self.settings_obj = T6_PSI_settings.load_obj()
        self.eq_obj = construct_eqn()
        self.state = initial_state
        self.T_init = T_init
        self.T_final = T_final
        self.alpha_temp = alpha_temp
        self.num_moves_per_step = num_moves_per_step
        self.current_map = (current_map * self.settings_obj.current_unit
                           ) / self.settings_obj.VDD
        self.IR_PENALTY = 32 
        self.CONG_PENALTY = 15 
        self.congestion_enabled = congestion_enabled

    def sim_anneal(self, all_templates, template_list, signal_cong):
        T = self.T_init
        Tvals = []
        while T > self.T_final:
            Tvals.append(T)
            T = self.cool_down(T, self.alpha_temp)
        cur_state = self.state
        template_util = np.zeros(len(all_templates))
        total_tracks = 0
        num_tracks_layer = {}
        for i, layer in enumerate(self.settings_obj.TECH_LAYERS):
            attributes = self.settings_obj.LAYERS[layer]
            dirs = attributes['direction']
            if dirs == 'V':
                if self.congestion_enabled == 1:
                    num_tracks_layer[layer] = math.floor(
                    self.settings_obj.WIDTH_REGION / attributes['t_spacing'])
                else: 
                    num_tracks_layer[layer] = 0
            else:
                if self.congestion_enabled == 1:
                    num_tracks_layer[layer] = math.floor(
                    self.settings_obj.LENGTH_REGION / attributes['t_spacing'])
                else: 
                    num_tracks_layer[layer] = 0
            if self.congestion_enabled == 1:
                total_tracks = total_tracks + num_tracks_layer[layer]
            else:
                total_tracks = 1

        template_util = {}
        for i, template_num in enumerate(all_templates):
            template_util[template_num] = self.pdn_util(template_list,
                                                        template_num,
                                                        num_tracks_layer,
                                                        total_tracks)

        cur_energy, max_drop, self.state = self.energy(cur_state,
                                                       self.current_map,
                                                       all_templates,
                                                       template_list,
                                                       template_util,
                                                       signal_cong)


        #for n, T in tqdm(enumerate(Tvals), total=len(Tvals)):
        for n, T in enumerate(Tvals):
            for i in range(self.num_moves_per_step):
                next_state,next_pos = self.move(cur_state, all_templates)
                next_energy,next_max_drop = self.delta_energy(cur_state, cur_energy,
                                                next_state, self.current_map,
                                                template_list, template_util,signal_cong)
                delta_cost = next_energy - cur_energy
                accepted = self.acceptMove(delta_cost, T)
                #print("delta_cost= %e T= %f accepted= %d"%(delta_cost,T,accepted))
                if accepted:
                    self.state = next_state
                    cur_energy = next_energy
                    cur_state = next_state
                    max_drop[next_pos] = next_max_drop
        return self.state[4], cur_energy, max_drop[4]

    def energy(self, state, current_map, all_templates, template_list,
               template_util,signal_cong):
        """Calculates the length of the route."""
        e = 0
        max_drop = self.settings_obj.VDD * np.ones(len(state))
        beta_ir = np.zeros(len(state))
        lambda_cong = np.zeros(len(state))
        temp_util_min = template_util[min(template_util, key=template_util.get)]
        temp_util_max = template_util[max(template_util, key=template_util.get)]
        new_state = np.zeros(len(state))
        tot_cong =np.zeros(len(state))
        temp_util =np.zeros(len(state))
        for n, template in enumerate(state):
            tot_cong[n] = signal_cong[n] + template_util[template]
            if self.congestion_enabled ==1:
                temp_util[n] = (template_util[template]-temp_util_min)/(
                                temp_util_max-temp_util_min)
            else:
                temp_util[n] =0
            y = math.floor(n / self.settings_obj.NUM_REGIONS_X)
            x = n % self.settings_obj.NUM_REGIONS_X
            regional_current, map_row = self.eq_obj.get_regional_current(
                current_map, x, y)
            template_obj = template_list[template]
            g_start = template_obj.start
            G = template_obj.G
            J = self.eq_obj.create_J(regional_current, template_obj)
            G, J = self.eq_obj.add_vdd_to_G_J(G, J, template_obj, 0)
            J = sparse_mat.dok_matrix(J)
            solution = self.eq_obj.solve_ir(G, J)
            bot = g_start[0]    #M1 is shape -6
            top = g_start[1]
            V = solution[int(bot):int(top)]
            max_drop[n] = max(self.settings_obj.VDD - V)
            if max_drop[n] < self.settings_obj.IR_DROP_LIMIT:
                beta_ir[n] = 0
            else:
                beta_ir[n] = self.IR_PENALTY
            if tot_cong[n] < 1:
                lambda_cong[n] = 0
            else:
                lambda_cong[n] = self.CONG_PENALTY
    #        if(n ==0 ):
    #            V_full = V.T
    #        else:
    #            V_full = np.vstack((V_full,V.T))
    #        J_map = J[int(bot):int(top)]
    #        J_map = J_map.todense()
    #        if(n ==0 ):
    #            J_full = J_map.T
    #        else:
    #            J_full = np.vstack((J_full,J_map.T))
    #        print("region %d template %d"%(n,temp_num))
    #    with open('V_map.csv', 'w') as outfile:
    #            np.savetxt(outfile,V_full,delimiter=',')
    #    with open('J_map.csv', 'w') as outfile:
    #        np.savetxt(outfile,J_full,delimiter=',')
        for n, template in enumerate(state):
            dr_norm = max_drop[n]/self.settings_obj.IR_DROP_LIMIT
            #e = e + tot_cong[n] + dr_norm + (beta_ir[n] * (dr_norm - 1)) + (
            #    lambda_cong[n] * (tot_cong[n] - 1))
            e = e + temp_util[n] +  (beta_ir[n] * (dr_norm - 1)) + (
                lambda_cong[n] * (tot_cong[n] - 1))
        return e, max_drop, new_state

    def delta_energy(self, state_old, cur_energy, state_new, current_map,
                     template_list, template_util,signal_cong):
        difference = np.array(np.array(state_new) - np.array(state_old))
        index = np.nonzero(difference)
        if sum(difference) == 0:
            print("WARNING SAME VALUE")
            return cur_energy
        index = int(index[0])
        old_template = state_old[index]
        new_template = state_new[index]
        y = math.floor(index / self.settings_obj.NUM_REGIONS_X)
        x = index % self.settings_obj.NUM_REGIONS_X
        regional_current, map_row = self.eq_obj.get_regional_current(
            current_map, x, y)
        temp_util_min = template_util[min(template_util, key=template_util.get)]
        temp_util_max = template_util[max(template_util, key=template_util.get)]
        #temp_util_min = min(template_util)
        #temp_util_max = max(template_util)
        #print("min max %f %f "%(temp_util_min,temp_util_max),template_util)
        template_obj = template_list[old_template]
        G_old = template_obj.G
        J_old = self.eq_obj.create_J(regional_current, template_obj)
        G_old, J_old = self.eq_obj.add_vdd_to_G_J(G_old, J_old, template_obj, 0)
        J_old = sparse_mat.dok_matrix(J_old)
        solution = self.eq_obj.solve_ir(G_old, J_old)
        g_start = template_obj.start
        tot_cong = template_util[old_template] +signal_cong[index]
        if self.congestion_enabled ==1:
            temp_util = (template_util[old_template]-temp_util_min)/(
                                temp_util_max-temp_util_min)
        else:
            temp_util = 0

        V = solution[int(g_start[0]):int(g_start[1])]
        max_drop = max(self.settings_obj.VDD - V)
        if max_drop < self.settings_obj.IR_DROP_LIMIT:
            beta_ir = 0
        else:
            beta_ir = self.IR_PENALTY
        if(tot_cong < 1):
            lambda_cong = 0
        else:
            lambda_cong = self.CONG_PENALTY
        
        dr_norm = max_drop/self.settings_obj.IR_DROP_LIMIT
        #const_energy = cur_energy - (tot_cong + dr_norm + beta_ir * 
        #    (dr_norm - 1) + lambda_cong*(tot_cong- 1))
        const_energy = cur_energy - (temp_util + beta_ir * 
            (dr_norm - 1) + lambda_cong*(tot_cong- 1))

        template_obj = template_list[new_template]
        G_new = template_obj.G
        J_new = self.eq_obj.create_J(regional_current, template_obj)
        G_new, J_new = self.eq_obj.add_vdd_to_G_J(G_new, J_new, template_obj, 0)
        J_new = sparse_mat.dok_matrix(J_new)
        solution = self.eq_obj.solve_ir(G_new, J_new)
        g_start = template_obj.start
        V = solution[int(g_start[0]):int(g_start[1])]
        max_drop = max(self.settings_obj.VDD - V)
        tot_cong = template_util[new_template] +signal_cong[index]

        if self.congestion_enabled ==1:
            temp_util = (template_util[new_template]-temp_util_min)/(
                                temp_util_max-temp_util_min)
        else:
            temp_util = 0
        if max_drop < self.settings_obj.IR_DROP_LIMIT:
            beta_ir = 0
        else:
            beta_ir = self.IR_PENALTY
        if(tot_cong < 1):
            lambda_cong = 0
        else:
            lambda_cong = self.CONG_PENALTY
        
        dr_norm = max_drop/self.settings_obj.IR_DROP_LIMIT
        #new_energy = const_energy + (tot_cong + dr_norm + beta_ir * 
        #    (dr_norm -1) + lambda_cong*(tot_cong- 1))
        #print("temp_util %f"%temp_util)
        new_energy = const_energy + (temp_util + beta_ir * 
            (dr_norm -1) + lambda_cong*(tot_cong- 1))
        return new_energy,max_drop

    def cool_down(self, T, alpha):
        T = alpha * T
        return T

    def acceptMove(self, delta_cost, T):
        k = 1.38064852 * (10**(0))
        if delta_cost < 0:
            accept = 1
        else:
            boltz_prob = math.exp((-1 * delta_cost) / (k * T))
            r = random.uniform(0, 1)
            #print("random value: %e, boltz_prob: %e"%(r,boltz_prob))
            if r < boltz_prob:
                accept = 1
            else:
                accept = 0
        return accept
    def move(self, state, all_templates):
        """Randomly pick a tempalte from the list of all tempaltes and insert
        in the current state.
        """
        flag = 0
        while flag == 0:
            a = random.randint(0, len(all_templates) - 1)
            b = random.randint(0, len(state) - 1)
            if state[b] != all_templates[a]:
                flag = 1
        state_new = list(state)
        state_new[b] = all_templates[a]
        return state_new,b

    def pdn_util(self, template_list, template_num, num_tracks_layer,
                 total_tracks):
        template_obj = template_list[template_num]
        util_layer = np.zeros(template_obj.pitches.shape[0])
        used_tracks = np.zeros(template_obj.pitches.shape[0])
        template_used_tracks = 0
        for i, layer in enumerate(self.settings_obj.PDN_layers_ids):
            attributes = self.settings_obj.LAYERS[layer]
            dirs = attributes['direction']
            if self.congestion_enabled == 1:
                tracks_per_stripe = math.ceil(template_obj.widths[i] /
                                          attributes['t_spacing'])
            else:
                tracks_per_stripe = 0
            if dirs == 'V':
                used_tracks[i] = math.floor(
                    (self.settings_obj.WIDTH_REGION * tracks_per_stripe) /
                    template_obj.pitches[i])
            else:
                used_tracks[i] = math.floor(
                    (self.settings_obj.LENGTH_REGION * tracks_per_stripe) /
                    template_obj.pitches[i])
            if self.congestion_enabled == 1:
                util_layer[i] = used_tracks[i] / num_tracks_layer[layer]
            else:
                util_layer[i] = 0
            template_used_tracks = template_used_tracks + used_tracks[i]
        template_util = (template_used_tracks / total_tracks) * 2
        return template_util
