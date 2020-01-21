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
Builds the GV = J set of equations and solves it.
@author:Vidya A Chhabria
Main code that integrates all submodules to generate training data. Creates the
baseline alogrithm for the ML model to learn by generating sample output data
for input imagess. It calls the functions to load the templates, create the
final G matrix and solve GV=P equation to calculate the worst case IR drop.
"""

import time
import math
import fractions as frac
from scipy import sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
import numpy as np
# import os.pat
from create_template_new import template
from create_template_new import node
from T6_PSI_settings import T6_PSI_settings

class construct_eqn():
    """ Class that creates the linear system of equations GV=J and solves it
    Attributes

    """

    def __init__(self):
        """ Initializes construct_eq class with the height and width of the
        region and the c4 bump location"""
        self.settings_obj = T6_PSI_settings.load_obj()
        self.size_region_x = int(self.settings_obj.WIDTH_REGION * 1e6)
        self.size_region_y = int(self.settings_obj.LENGTH_REGION * 1e6)
        self.WIDTH_REGION = self.settings_obj.WIDTH_REGION* self.settings_obj.lef_unit 
        self.LENGTH_REGION = self.settings_obj.LENGTH_REGION * self.settings_obj.lef_unit
        self.VDD = self.settings_obj.VDD
        self.NUM_REGIONS_X = self.settings_obj.NUM_REGIONS_X
        self.NUM_REGIONS_Y = self.settings_obj.NUM_REGIONS_Y
        self.c4_bump_loc = self.c4_bump_dist_regional(self.settings_obj.n_c4)

    def c4_bump_dist_regional(self, n_c4):
        """ Defines the C4 bump distribution across the chip
        Args:
            n_c4: Number of C4 bumps to distribute uniformly
        Returns:
            c4_bump_det: A n_c4X3 matrix which contains x-coordinate
            y-coordinate and VDD value for every bump in a row of the matrix
        """
        n_vdd_reg = int(
            np.floor(n_c4 / (self.NUM_REGIONS_X * self.NUM_REGIONS_Y)))
        n_vdd_1d = int(np.floor(n_vdd_reg**0.5))
        c4_bump_det = np.zeros((n_vdd_reg, 3))
        count = 0
        for i in range(0, n_vdd_1d):
            for j in range(0, n_vdd_1d):
                c4_bump_det[count, :] = [
                    ((i + 0.5) * self.WIDTH_REGION) / n_vdd_1d,
                    ((j + 0.5) * self.LENGTH_REGION) / n_vdd_1d, self.VDD
                ]
                count = count + 1
        return c4_bump_det

    def get_regional_voltage(self, template_obj,V, ind_x, ind_y):
        xcor = int((ind_x * self.WIDTH_REGION))
        ycor = int((ind_y * self.LENGTH_REGION))
        region_voltage = np.zeros((0, 3))
        for i in range(V.shape[0]- self.c4_bump_loc.shape[0]):
            node_h = template_obj.get_node_from_loc(i)
            if node_h.get_layer() == 0:
                l,x,y = node_h.get_loc()
                voltage_set = np.zeros((1,3))
                voltage_set[0,0] = (x+xcor)/self.settings_obj.lef_unit
                voltage_set[0,1] = (y+ycor)/self.settings_obj.lef_unit
                voltage_set[0,2] = V[i]
                region_voltage = np.append(region_voltage,voltage_set,axis=0)
        return region_voltage
                
    def get_regional_current(self, chip_current, ind_x, ind_y):
        """ Extracts current on a regional basis from chip current
        Args:
            chip_current: Current of the entire chip from current map
            ind_x: Index in X direction for regional current extraction
            ind_y: Index in Y direction for regional current extraction
        Returns:
            region_current: A 2D matrix which represents the current map for a
                            region
            current_row: A 1D row vector representing the current map of region
                         to save into data format needed for ML model
        """
        xcor = int((ind_x * self.size_region_x))
        ycor = int((ind_y * self.size_region_y))
        end_xcor = int(xcor + self.size_region_x)
        end_ycor = int(ycor + self.size_region_y)
        current_dis = chip_current[xcor:end_xcor, ycor:end_ycor]
        current_row = current_dis.reshape(-1)
        k = 0
        region_current = np.zeros(
            (current_dis.shape[0] * current_dis.shape[1], 3))
        # Create the 2D matrix (n_c4X3) which has x-cor, y-cor, and data
        for i in range(current_dis.shape[0]):
            for j in range(current_dis.shape[1]):
                region_current[k, 0] = np.round(
                    i * (self.WIDTH_REGION / current_dis.shape[0]))
                region_current[k, 1] = np.round(
                    j * (self.LENGTH_REGION / current_dis.shape[1]))
                region_current[k, 2] = current_dis[i, j]
                k = k + 1
        return region_current, current_row


    def create_J(self,template_obj, regional_cur):
    
        """ Creates the current density vector for a region, J
        This function also sets offsets to ensure no current source lies on the
        edge or beyond the die boundary. So that every current source is
        connected to a stripe and not floating.
        Args:
            regional_cur: 2D current map for the region
            template_obj: Template object encapsulates the pysical attributes
                          of the template such as charateristics of the PDN.
        Returns:
            J: current density column vector
        """
        J = np.zeros([template_obj.num_nodes, 1])
        #TODO store it in the node??? or keep tamplate clean
        for i in range(regional_cur.shape[0]):
            node_h = template_obj.get_node(0,regional_cur[i, 0],regional_cur[i, 1])
            node_loc = node_h.get_G_loc()
            J[node_loc, 0] += -regional_cur[i, 2]
        return J

    def add_vdd_to_G_J(self,J, template_obj):
        """ Adds the additional row and column for a type 2 element stamp
        This function add the additional row and column to the conductance
        matrix, G, to account for the voltage sources and the additional rows
        to the current density matrix, J, for the voltage source
        Args:
            template_obj: Template object encapsulates the pysical attributes
                          of the template such as charateristics of the PDN.
            template_start: To have the location of the beginning of every
                            template in G
            G: The 2D conduction matrix which needs to be modified with the
               additional rows and columns to account for the voltage source
            J: The column vector which needs to be modified with the additional
               row to account for the voltage source
        Returns:
            G: Updated conductance matrix
            J: Updated current density vector
        """
        G = template_obj.get_G_mat();
        vdds = np.zeros((G.shape[0], (self.c4_bump_loc).shape[0]))
        for i in range(self.c4_bump_loc.shape[0]):
            node_h = template_obj.get_nearest_node(template_obj.num_layers-1, self.c4_bump_loc[i,0] , self.c4_bump_loc[i, 1])
            loc = node_h.get_G_loc();
            vdds[loc, i] = 1 
        addn_zeros = sparse_mat.dok_matrix( np.zeros( 
            ((self.c4_bump_loc).shape[0], (self.c4_bump_loc).shape[0])  ) )
        vdds = sparse_mat.dok_matrix(vdds)
        G_out = sparse_mat.bmat([[G, vdds], [np.transpose(vdds), addn_zeros]])
        J_add_vdds = self.VDD * np.ones([self.c4_bump_loc.shape[0], 1])
        # Adding the additional rows to the J vector
        J_out = np.concatenate((J, J_add_vdds), axis=0)
        return G_out, J_out

    def solve_ir(self, G_in, J_in):
        """ Solves the system of linear equations GV=J, to find V
        This function uses sparse matrix solver with umfpack
        Args:
            G_in: The 2D conduction matrix
            J_in: The current denisty vector
        Returns:
            V: Voltage at every node
        """
        G = sparse_mat.dok_matrix.tocsc(G_in)
        I = sparse_mat.identity(G.shape[0]) * 1e-13
        G = G + I
        J = sparse_mat.dok_matrix.tocsc(J_in)
        V = sparse_algebra.spsolve(G, J, permc_spec=None, use_umfpack=True)
        return V

if __name__ == '__main__':
    print("Not designed to be run alone")
    pass
