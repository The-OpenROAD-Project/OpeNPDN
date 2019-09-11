#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the GV = J set of equations and solves it.
@author: chhab011
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
# import os.path
from template_construction import template_def
from create_template import define_templates
from T6_PSI_settings import T6_PSI_settings

class construct_eqn():
    """ Class that creates the linear system of equations GV=J and solves it
    Attributes

    """

    def __init__(self):
        """ Initializes construct_eq class with the height and width of the
        region and the c4 bump location"""
        self.settings_obj = T6_PSI_settings()
        self.size_region_x = int(self.settings_obj.WIDTH_REGION * 1e6)
        self.size_region_y = int(self.settings_obj.LENGTH_REGION * 1e6)
        self.WIDTH_REGION = self.settings_obj.WIDTH_REGION
        self.LENGTH_REGION = self.settings_obj.LENGTH_REGION
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
                region_current[k, 0] = (
                    i * (self.WIDTH_REGION / current_dis.shape[0]))
                region_current[k, 1] = (
                    j * (self.LENGTH_REGION / current_dis.shape[1]))
                region_current[k, 2] = current_dis[i, j]
                k = k + 1
        return region_current, current_row

    def connect_resistors(self, G, loc1, loc2, cond):
        """ Connects adjacent conductances to form the conductance matrix
        Args:
            G: 2D conductance matrix
            loc1: Row location to insert value in the conduction matrix
            loc2: Column location to insert value in the conduction matrix
            cond: Value of conductance between the two nodes
        Returns:
            G: Updated 2D conductance matrix
        """
        G[loc1, loc2] = -cond
        G[loc2, loc1] = -cond
        G[loc1, loc1] = G[loc1, loc1] + cond
        G[loc2, loc2] = G[loc2, loc2] + cond
        return G

    def create_J(self, regional_cur, template_obj):
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
        dir_bot = template_obj.dirs[0]    # Direction of bottom layer
        if dir_bot == 0:    # if lower most layer is horizontal,
            #set the vertical pitch
            pitch_vert_bottom = template_obj.pitches[0]
            offset_vert_bottom = template_obj.offset[0]
            for dir_num, direction in enumerate(
                    template_obj.dirs):    #find first vertical layer
                if direction == 1:    # first vert layer
                    break
            pitch_hor_bottom = template_obj.pitches[dir_num]
            offset_hor_bottom = template_obj.offset[dir_num]
        else:    #vert layer so we set horizontal pitch
            pitch_hor_bottom = template_obj.pitches[0]
            offset_hor_bottom = template_obj.offset[0]
            for dir_num, direction in enumerate(
                    template_obj.dirs):    #find first horizontal layer
                if direction == 1:    # first horz layer
                    break
            pitch_vert_bottom = template_obj.pitches[dir_num]
            offset_vert_bottom = template_obj.offset[dir_num]
        # Find the current source position, i.e., x and y coordinate for each
        # current source along with the value
        node_cs = np.column_stack(((pitch_hor_bottom/template_obj.xpitch)* \
                  (np.floor(regional_cur[:, 0]/pitch_hor_bottom))+offset_hor_bottom, \
                  (pitch_vert_bottom/template_obj.ypitch)* \
                  (np.floor(regional_cur[:, 1]/pitch_vert_bottom))+offset_vert_bottom, \
                  (regional_cur[:, 2])))
        max_x = max(node_cs[node_cs[:, 0] < template_obj.num_x, 0])
        max_y = max(node_cs[node_cs[:, 1] < template_obj.num_y, 1])
        node_cs[node_cs[:, 0] > max_x, 0] = max_x
        node_cs[node_cs[:, 1] > max_y, 1] = max_y
        J = np.zeros([(template_obj.G).shape[0], 1])
        for i in range(node_cs.shape[0]):
            node = math.floor(
                template_def.convert_index(template_obj, node_cs[i, 0],
                                           node_cs[i, 1]))
            J[node, 0] += -node_cs[i, 2]
        return J

    def add_vdd_to_G_J(self, G, J, template_obj, template_start):
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
        pitch_top = template_obj.pitches[(template_obj.pitches).shape[0] - 1]
        dir_top = template_obj.dirs[(template_obj.dirs).shape[0] - 1]
        offset_top = template_obj.offset[(template_obj.offset).shape[0] - 1]
        if dir_top == 0:    # top metal is horizontal
            # Create the matrix of Vdd and nodes, i.e., a matrix that contains
            # x, y cordimnates and the voltage value
            node_vdd = np.column_stack(((np.floor(self.c4_bump_loc[:, 0]/ \
                       template_obj.xpitch)), (pitch_top/template_obj.ypitch)* \
                       (np.floor(self.c4_bump_loc[:, 1]/pitch_top))+offset_top))
        else:    # top metal is veritcal
            # Create the matrix of Vdd and nodes, i.e., a matrix that contains
            # x, y cordimnates and the voltage value
            node_vdd = np.column_stack((((pitch_top/template_obj.xpitch)* \
                       (np.floor(self.c4_bump_loc[:, 0]/pitch_top)))+offset_top, \
                       (np.floor(self.c4_bump_loc[:, 1]/template_obj.ypitch))))
        vdds = np.zeros((G.shape[0], (self.c4_bump_loc).shape[0]))
        g9_start = template_start + template_obj.start[
            (template_obj.start).shape[0] - 1]
        # Adding the additional rows to the G vector
        for i in range(node_vdd.shape[0]):
            vdds[(g9_start + math.floor(
                template_def.
                convert_index(template_obj, node_vdd[i, 0], node_vdd[i, 1]))
                 ), i] = 1    #self.vdd
        addn_zeros = sparse_mat.dok_matrix(
            np.zeros(
                ((self.c4_bump_loc).shape[0], (self.c4_bump_loc).shape[0])))
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

    def create_G_J(self, templates, chip_current, template_list):
        """ Constructs the conductance matrix and the current density matrix for
        the entire chip
        Args:
            templates: A 1D array of all the templates used in the PDN for the
            entire chip
            chip_current: Entire current map across the chip
            template_list: List ofall the template_obj for all the possible
            templates irrespective of their used in the PDN
        Returns:
            G: Entire conductance matrix across the chip
            J: Entire current density vector across the chip
            template_start: the location in the G matrix where a new template
            begins for the all the neighboring regions
        """
        template_start = np.zeros(len(templates), 'int')
        # Converting  templates to represent a 2D map
        templates_2d = np.reshape(templates,
                                  (self.NUM_REGIONS_Y, self.NUM_REGIONS_X))
        for i in range(self.NUM_REGIONS_X * self.NUM_REGIONS_Y):
            image_count_x = i % self.NUM_REGIONS_X
            image_count_y = int(i / self.NUM_REGIONS_X)
            regional_current, current_row = self.get_regional_current(
                chip_current, image_count_x, image_count_y)
            G_single = template_list[templates[i]].G
            J_single = self.create_J(regional_current,
                                     template_list[templates[i]])
            if i == 0:
                G = G_single
                J = J_single
            else:
                G = sparse_mat.block_diag((G, G_single), format='dok')
                J = np.concatenate((J, J_single), axis=0)
            template_obj = template_list[templates[i]]
            if i + 1 < self.NUM_REGIONS_X * self.NUM_REGIONS_Y:
                template_start[i + 1] = G.shape[0]

        # This block creates connections between templates in adjacent regions
        for y in range(self.NUM_REGIONS_Y):
            for x in range(self.NUM_REGIONS_X):
                region_num = self.NUM_REGIONS_X * y + x
                template_obj = template_list[templates_2d[y, x]]
                #print("region %d"%region_num)
                if (x < self.NUM_REGIONS_X -
                        1):    # Find the template to the right
                    template_obj_right = template_list[templates_2d[y, x + 1]]
                if (y < self.NUM_REGIONS_Y -
                        1):    # Find the template at the bottom
                    template_obj_below = template_list[templates_2d[y + 1, x]]

                for k in range(template_obj.n_layers):
                    pitch1 = template_obj.pitches[k]
                    g_start1 = template_obj.start[k]
                    # Make the horziontal connections
                    if (template_obj.dirs[k] == 0
                            and x < self.NUM_REGIONS_X - 1):    # horizontal
                        pitch2 = template_obj_right.pitches[k]
                        g_start2 = template_obj_right.start[k]
                        x_node1 = template_obj.num_x - 1
                        x_node2 = 0
                        grid1 = template_obj.ypitch
                        grid2 = template_obj_right.ypitch
                        offset1 = template_obj.offset[k]
                        offset2 = template_obj_right.offset[k]
                        grid = frac.gcd(grid1, grid2)
                        y_node1 = 0
                        y_node2 = 0
                        cond = template_obj.widths[k] / (
                            template_obj.rho_s[k] * grid
                        )    # assume minimum lenght of grid
                        # Handle offset at the edge of the chip to ensure that
                        # no stripe is on the edge or beyond the chip area
                        #print("layer %d right "%k)
                        #print("offsets %d %d pitches %3.1f %3.1f"%(offset1,offset2,pitch1/grid1,pitch2/grid2))
                        while (y_node1 * grid1 < self.LENGTH_REGION
                               and y_node2 * grid2 < self.LENGTH_REGION):
                            if (((np.round((y_node1-offset1)*grid1*1e9) % \
                                np.round(pitch1*1e9)) < 1) and
                                ((np.round((y_node2-offset2)*grid2*1e9) % \
                                np.round(pitch2*1e9)) < 1)):
                                if (abs(y_node1 * grid1 - y_node2 * grid2) <
                                        1e-10):
                                    loc1 = template_start[region_num] + \
                                           g_start1 + template_obj.convert_index(x_node1, y_node1)
                                    loc2 = template_start[region_num+1] + \
                                           g_start2 + template_obj_right.convert_index(x_node2, y_node2) # to the right
                                    self.connect_resistors(G, loc1, loc2, cond)
                                    #print("connection at y nodes %f %f at grids %f %f"%( 
                                    #            y_node1, y_node2, grid1, grid2))
                                    #print("connect at loc %d %d"%( 
                                    #    loc1-g_start1,loc2-g_start2))
                            if y_node1 * grid1 < y_node2 * grid2:
                                y_node1 += 1
                            else:
                                y_node2 += 1
                    elif (template_obj.dirs[k] == 0
                          and x == self.NUM_REGIONS_X - 1):    # horizontal
                        pass
                    # Make the vertical connection
                    elif (template_obj.dirs[k] == 1
                          and y < self.NUM_REGIONS_Y - 1):    # vertical
                        #print("layer %d bottom "%k)
                        pitch2 = template_obj_below.pitches[k]
                        g_start2 = template_obj_below.start[k]
                        grid1 = template_obj.xpitch
                        grid2 = template_obj_below.xpitch
                        y_node1 = template_obj.num_y - 1
                        y_node2 = 0
                        offset1 = template_obj.offset[k]
                        offset2 = template_obj_below.offset[k]
                        grid = frac.gcd(grid1 * 1e9, grid2 * 1e9) * 1e-9
                        x_node1 = 0
                        x_node2 = 0
                        cond = template_obj.widths[k] / (
                            template_obj.rho_s[k] * grid
                        )    # assume minimum lenght of grid
                        # Handle offset at the edge of the chip to ensure that
                        # no stripe is on the edge or beyond the chip area
                        #print("offsets %d %d pitches %3.1f %3.1f"%(offset1,offset2,pitch1/grid1,pitch2/grid2))
                        while (x_node1 * grid1 < self.WIDTH_REGION
                               and x_node2 * grid2 < self.WIDTH_REGION):
                            if (((np.round((x_node1-offset1)*grid1*1e9) % \
                                np.round(pitch1*1e9)) < 1) and \
                                ((np.round((x_node2-offset2)*grid2*1e9) % \
                                np.round(pitch2*1e9)) < 1)):
                                if (abs(x_node1 * grid1 - x_node2 * grid2) <
                                        1e-10):    # grids are aligned
                                    loc1 = template_start[region_num] + \
                                           g_start1 + template_obj.convert_index(x_node1, y_node1)
                                    loc2 = template_start[region_num+self.NUM_REGIONS_X] + \
                                           g_start2 + template_obj_below.convert_index(x_node2, y_node2) # below
                                    self.connect_resistors(G, loc1, loc2, cond)
                                    #print("connection at y nodes %f %f at grids %f %f"%( 
                                    #            x_node1, x_node2, grid1, grid2))
                                    #print("connect at loc %d %d"%( 
                                    #    loc1-g_start1,loc2-g_start2))
                            if x_node1 * grid1 < x_node2 * grid2:
                                x_node1 += 1
                            else:
                                x_node2 += 1
                    elif (template_obj.dirs[k] == 1
                          and y == self.NUM_REGIONS_Y - 1):    # vertical
                        pass
                    else:
                        print(
                            "ERROR in condition please check the source code region grids are not stiched"
                        )
        for n, cur_template in enumerate(templates):
            template_obj = template_list[cur_template]
            G, J = self.add_vdd_to_G_J(G, J, template_obj, template_start[n])
        J = sparse_mat.dok_matrix(J)
        return G, J, template_start


if __name__ == '__main__':
    settings_obj = T6_PSI_settings()
    all_template_list = define_templates(settings_obj, generate_g=1)
    start = time.time()
    eq_obj = construct_eqn()
    print("Processing Current Maps")
    map_start = 1
    num_maps = 15
    input_image = settings_obj.map_dir + "current_map_%d.csv" % (
            + map_start)
    currents = np.genfromtxt(input_image, delimiter=',')
    max_drop = settings_obj.VDD
    templates_used = [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]
    G, J, template_start = eq_obj.create_G_J(templates_used, currents,
                                      all_template_list)
    solution = eq_obj.solve_ir(G, J)
    max_drop = 0
    for n, template in enumerate(templates_used):
        g_start = all_template_list[template].start
        bot = g_start[g_start.shape[0] - 6]
        top = g_start[g_start.shape[0] - 5]
        V = solution[int(template_start[n] + bot):int(template_start[n] + top)]
        max_drop_temp = max(settings_obj.VDD - V)
        max_drop = max([max_drop, max_drop_temp])
        if n == 0:
            V_full = V.T
        else:
            V_full = np.vstack((V_full, V.T))
    with open('V_map.csv', 'w') as outfile:
        np.savetxt(outfile, V_full, delimiter=',')

    end = time.time()
    print("solving time:", end - start)
    print("\n")
